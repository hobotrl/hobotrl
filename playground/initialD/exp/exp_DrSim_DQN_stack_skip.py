# -*- coding: utf-8 -*-
"""Experiment script for DQN-based lane decision.

Author: Jingchu Liu
"""
# Basics
import os
import signal
import time
import sys
import traceback
from collections import deque
import numpy as np
# Tensorflow
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer
sys.path.append('../../..')
sys.path.append('..')
# Hobotrl
import hobotrl as hrl
from hobotrl.environments import FrameStack
from hobotrl.sampling import TransitionSampler
from hobotrl.playback import BalancedMapPlayback
# initialD
# from ros_environments.honda import DrivingSimulatorEnv
from ros_environments.clients import DrivingSimulatorEnvClient as DrivingSimulatorEnv
# Gym
from gym.spaces import Discrete, Box

# Environment
def func_compile_reward(rewards):
    return rewards

def func_compile_obs(obss):
    obs1 = obss[-1][0]
    print obss[-1][1]
    print obs1.shape
    return obs1

ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
def func_compile_action(action):
    ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
    return ACTIONS[action]

def func_compile_exp_agent(state, action, rewards, next_state, done):
    global momentum_ped
    global momentum_opp
    global ema_speed

    # Compile reward
    rewards = np.mean(np.array(rewards), axis=0)
    rewards = rewards.tolist()
    rewards.append(np.logical_or(action==1, action==2))
    print (' '*10+'R: ['+'{:4.2f} '*len(rewards)+']').format(*rewards),

    speed = rewards[2]
    ema_speed = 0.5*ema_speed + 0.5*speed
    longest_penalty = rewards[1]
    obs_risk = rewards[5]
    momentum_opp = (rewards[3]<0.5)*(momentum_opp+(1-rewards[3]))
    momentum_opp = min(momentum_opp, 20)

    # obstacle
    rewards[0] *= 0.0
    # distance to
    rewards[1] *= -10.0*(rewards[1]>2.0)
    # velocity
    rewards[2] *= 10
    # opposite
    rewards[3] = -20*(0.9+0.1*momentum_opp)*(momentum_opp>1.0)
    # ped
    momentum_ped = (rewards[4]>0.5)*(momentum_ped+rewards[4])
    momentum_ped = min(momentum_ped, 12)
    rewards[4] = -40*(0.9+0.1*momentum_ped)*(momentum_ped>1.0)
    # obs factor
    rewards[5] *= -100.0
    # steering
    rewards[6] *= -10.0
    reward = np.sum(rewards)/100.0
    print '{:6.4f}, {:6.4f}'.format(momentum_opp, momentum_ped),
    print ': {:7.4f}'.format(reward)

    # early stopping
    if ema_speed < 0.1:
        if longest_penalty > 0.5:
            print "[Early stopping] stuck at intersection."
            done = True
        if obs_risk > 0.1:
            print "[Early stopping] stuck at obstacle."
            done = True
        if momentum_ped>1.0:
            print "[Early stopping] stuck on pedestrain."
            done = True


    return state, action, reward, next_state, done

def gen_backend_cmds():
    # ws_path = '/home/lewis/Projects/catkin_ws_pirate03_lowres350/'
    ws_path = '/Projects/catkin_ws/'
    # initialD_path = '/home/lewis/Projects/hobotrl/playground/initialD/'
    initialD_path = '/Projects/hobotrl/playground/initialD/'
    backend_path = initialD_path + 'ros_environments/backend_scripts/'
    utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
    backend_cmds = [
        # 1. Parse maps
        ['python', utils_path+'parse_map.py',
         ws_path+'src/Map/src/map_api/data/honda_wider.xodr',
         utils_path+'road_segment_info.txt'],
        # 2. Generate obs and launch file
        ['python', utils_path+'gen_launch_dynamic.py',
         utils_path+'road_segment_info.txt', ws_path,
         utils_path+'honda_dynamic_obs_template.launch', 30],
        # 3. start roscore
        ['roscore'],
        # 4. start reward function script
        ['python', backend_path+'gazebo_rl_reward.py'],
        # ['python', backend_path+'rl_reward_function.py'],
        # 5. start simulation restarter backend
        ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
        # 6. [optional] video capture
        ['python', backend_path+'non_stop_data_capture.py', 0]
    ]
    return backend_cmds

env = DrivingSimulatorEnv(
    address="10.31.40.197", port='6003',
    # address='localhost', port='6003',
    backend_cmds=gen_backend_cmds(),
    defs_obs=[
        ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
        ('/decision_result', 'std_msgs.msg.Int16')
    ],
    defs_reward=[
        ('/rl/has_obstacle_nearby', 'std_msgs.msg.Bool'),
        ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
        ('/rl/car_velocity', 'std_msgs.msg.Float32'),
        ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
        ('/rl/on_pedestrian', 'std_msgs.msg.Bool'),
        ('/rl/obs_factor', 'std_msgs.msg.Float32'),
    ],
    defs_action=[('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')],
    rate_action=10.0,
    window_sizes={'obs': 2, 'reward': 3},
    buffer_sizes={'obs': 2, 'reward': 3},
    func_compile_obs=func_compile_obs,
    func_compile_reward=func_compile_reward,
    func_compile_action=func_compile_action,
    step_delay_target=0.5)
# TODO: define these Gym related params insode DrivingSimulatorEnv
env.observation_space = Box(low=0, high=255, shape=(350, 350, 3))
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
env.action_space = Discrete(len(ACTIONS))
env = FrameStack(env, 3)

# Agent
def f_net(inputs):
    inputs = inputs[0]
    inputs = inputs/128 - 1.0
    # (640, 640, 3*n) -> ()
    with tf.device('/gpu:0'):
        conv1 = layers.conv2d(
            inputs=inputs, filters=16, kernel_size=(8, 8), strides=1,
            kernel_regularizer=l2_regularizer(scale=1e-2),
            activation=tf.nn.relu, name='conv1')
        print conv1.shape
        pool1 = layers.max_pooling2d(
            inputs=conv1, pool_size=3, strides=4, name='pool1')
        print pool1.shape
        conv2 = layers.conv2d(
            inputs=pool1, filters=16, kernel_size=(5, 5), strides=1,
            kernel_regularizer=l2_regularizer(scale=1e-2),
            activation=tf.nn.relu, name='conv2')
        print conv2.shape
        pool2 = layers.max_pooling2d(
            inputs=conv2, pool_size=3, strides=3, name='pool2')
        print pool2.shape
        conv3 = layers.conv2d(
             inputs=pool2, filters=64, kernel_size=(3, 3), strides=1,
             kernel_regularizer=l2_regularizer(scale=1e-2),
             activation=tf.nn.relu, name='conv3')
        print conv3.shape
        pool3 = layers.max_pooling2d(
            inputs=conv3, pool_size=3, strides=2, name='pool3',)
        print pool3.shape
        depth = pool3.get_shape()[1:].num_elements()
        inputs = tf.reshape(pool3, shape=[-1, depth])
        print inputs.shape
        hid1 = layers.dense(
            inputs=inputs, units=256, activation=tf.nn.relu,
            kernel_regularizer=l2_regularizer(scale=1e-2), name='hid1')
        print hid1.shape
        hid2 = layers.dense(
            inputs=hid1, units=256, activation=tf.nn.relu,
            kernel_regularizer=l2_regularizer(scale=1e-2), name='hid2')
        print hid2.shape
        q = layers.dense(
            inputs=hid2, units=len(ACTIONS), activation=None,
            kernel_regularizer=l2_regularizer(scale=1e-2), name='q')
    return {"q": q}

optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
target_sync_rate = 1e-3
state_shape = env.observation_space.shape
graph = tf.get_default_graph()
global_step = tf.get_variable(
    'global_step', [], dtype=tf.int32,
    initializer=tf.constant_initializer(0), trainable=False)

agent = hrl.DQN(
    f_create_q=f_net, state_shape=state_shape,
    # OneStepTD arguments
    num_actions=len(ACTIONS), discount_factor=0.9, ddqn=False,
    # target network sync arguments
    target_sync_interval=1,
    target_sync_rate=target_sync_rate,
    # epsilon greeedy arguments
    greedy_epsilon=0.2,
    # optimizer arguments
    network_optimizer=hrl.network.LocalOptimizer(optimizer_td, 10.0),
    # max_gradient=10.0,
    # sampler arguments
    sampler=TransitionSampler(BalancedMapPlayback(
        num_actions=len(ACTIONS), capacity=15000),
        batch_size=8, interval=1),
    # checkpoint
    global_step=global_step)


def log_info(update_info):
    global action_fraction
    global action_td_loss
    global agent
    global next_state
    global ACTIONS
    global n_steps
    global done
    global cum_td_loss
    global cum_reward
    global n_ep
    global exploration_off
    global cnt_skip
    global n_skip
    global t_learn
    global t_infer
    summary_proto = tf.Summary()
    # modify info dict keys
    k_del = []
    new_info = {}
    for k, v in update_info.iteritems():
        if 'FitTargetQ' in k:
            new_info[k[14:]] = v  # strip "FitTargetQ\td\"
            k_del.append(k)
    update_info.update(new_info)
    if 'td_losses' in update_info:
        prt_str = zip(
            update_info['action'], update_info['q'],
            update_info['target_q'], update_info['td_losses'],
            update_info['reward'], update_info['done'])
        for s in prt_str:
            action_fraction *= 0.9
            action_fraction[s[0]] += 0.1
            action_td_loss[s[0]] = 0.9*action_td_loss[s[0]] + 0.1*s[3]
            if cnt_skip==n_skip:
                pass
                # print ("{} "+"{:8.5f} "*4+"{}").format(*s)
        # print action_fraction
        # print action_td_loss
    for tag in update_info:
        summary_proto.value.add(
            tag=tag, simple_value=np.mean(update_info[tag]))
    if 'q' in update_info and \
       update_info['q'] is not None:
        q_vals = update_info['q']
        summary_proto.value.add(
            tag='q_vals_max',
            simple_value=np.max(q_vals))
        summary_proto.value.add(
            tag='q_vals_min',
            simple_value=np.min(q_vals))
    if cnt_skip == n_skip:
        next_q_vals_nt = agent.learn_q(np.asarray(next_state)[np.newaxis, :])[0]
        for i, ac in enumerate(ACTIONS):
            summary_proto.value.add(
                tag='next_q_vals_{}'.format(ac[0]),
                simple_value=next_q_vals_nt[i])
            summary_proto.value.add(
                tag='next_q_vals_{}'.format(ac[0]),
                simple_value=next_q_vals_nt[i])
            summary_proto.value.add(
                tag='action_td_loss_{}'.format(ac[0]),
                simple_value=action_td_loss[i])
            summary_proto.value.add(
                tag='action_fraction_{}'.format(ac[0]),
                simple_value=action_fraction[i])
        p_dict = sorted(zip(
            map(lambda x: x[0], ACTIONS), next_q_vals_nt))
        max_idx = np.argmax([v for _, v in p_dict])
        p_str = "({:.3f}) ({:3d})[Q_vals]: ".format(
            time.time(), n_steps)
        for i, (a, v) in enumerate(p_dict):
            if a == ACTIONS[next_action][0]:
                sym = '|x|' if i==max_idx else ' x '
            else:
                sym = '| |' if i==max_idx else '   '
            p_str += '{}{:3d}: {:.4f} '.format(sym, a, v)
        print p_str

    cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info \
        and update_info['td_loss'] is not None else 0
    cum_reward += reward

    if done:
        print ("Episode {} done in {} steps, reward is {}, "
               "average td_loss is {}. No exploration {}").format(
                   n_ep, n_steps, cum_reward,
                   cum_td_loss/n_steps, exploration_off)
        if not exploration_off:
            summary_proto.value.add(
                tag='num_episode', simple_value=n_ep)
            summary_proto.value.add(
                tag='cum_reward', simple_value=cum_reward)
            summary_proto.value.add(
                tag='per_step_reward', simple_value=cum_reward/n_steps)
        else:
            summary_proto.value.add(
                tag='num_episode_noexplore', simple_value=n_ep)
            summary_proto.value.add(
                tag='cum_reward_noexplore',
                simple_value=cum_reward)
            summary_proto.value.add(
                tag='per_step_reward_noexplore', simple_value=cum_reward/n_steps)

    return summary_proto

n_interactive = 0
n_skip = 8
n_additional_learn = 4
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)

try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(
        graph=tf.get_default_graph(),
        is_chief=True,
        init_op=tf.global_variables_initializer(),
        logdir='./experiment',
        save_summaries_secs=10,
        save_model_secs=600)

    with sv.managed_session(config=config) as sess:
        agent.set_session(sess)
        action_fraction = np.ones(len(ACTIONS),) / (1.0*len(ACTIONS))
        action_td_loss = np.zeros(len(ACTIONS),)
        momentum_opp = 0.0
        momentum_ped = 0.0
        ema_speed = 10.0
        while True:
            n_ep += 1
            env.env.n_ep = n_ep  # TODO: do this systematically
            exploration_off = (n_ep%n_test==0)
            learning_off = exploration_off
            n_steps = 0
            cnt_skip = n_skip
            skip_reward = 0
            cum_td_loss = 0.0
            cum_reward = 0.0
            state  = env.reset()
            # action = 0  # default no-op at start
            action = agent.act(state, exploration=not exploration_off)
            skip_action = action
            while True:
                n_steps += 1
                # Env step
                next_state, reward, done, info = env.step(skip_action)
                state, action, reward, next_state, done = func_compile_exp_agent(
                    state, action, reward, next_state, done)
                skip_reward += reward
                # agent step
                cnt_skip -= 1
                update_info = {}
                t_learn, t_infer = 0, 0
                if cnt_skip==0 or done:
                    skip_reward /= (n_skip - cnt_skip)
                    if not learning_off:
                        t = time.time()
                        update_info = agent.step(
                            sess=sess, state=state, action=action,
                            reward=skip_reward, next_state=next_state,
                            episode_done=done)
                        t_learn += time.time() - t
                    t = time.time()
                    next_action = agent.act(next_state, exploration=not exploration_off)
                    t_infer += time.time() - t
                    cnt_skip = n_skip
                    skip_reward = 0
                    state, action = next_state, next_action  # s',a' -> s,a
                    skip_action = next_action
                else:
                    if not learning_off:
                        t = time.time()
                        update_info = agent.reinforce_(
                            sess=sess, state=None, action=None,
                            reward=None, next_state=None,
                            episode_done=None)
                        t_learn += time.time() - t
                    skip_action = 3  # no op during skipping
                sv.summary_computed(sess, summary=log_info(update_info))
                # addtional learning steps
                if not learning_off:
                    t = time.time()
                    for _ in range(n_additional_learn):
                        update_info = agent.reinforce_(
                            sess=sess, state=None, action=None,
                            reward=None, next_state=None,
                            episode_done=None)
                    t_learn += time.time() - t
                # print "Agent step learn {} sec, infer {} sec".format(t_learn, t_infer)
                if done:
                    break
except Exception as e:
    print e.message
    traceback.print_exc()
finally:
    print "="*30
    print "="*30
    print "Tidying up..."
    # kill orphaned monitor daemon process
    env.env.exit()
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    print "="*30

