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
from hobotrl.playback import BalancedMapPlayback, BigPlayback
from hobotrl.async import AsynchronousAgent
from hobotrl.utils import CappedLinear
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
    obs2 = obss[-2][0]
    obs3 = obss[-3][0]
    print obss[-1][1]
    # cast as uint8 is important otherwise float64
    obs = ((obs1 + obs2)/2).astype('uint8')
    # print obs.shape
    return obs.copy()

ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
AGENT_ACTIONS = ALL_ACTIONS[:3]
TIME_STEP_SCALES = [1, 2, 4, 8]


# AGENT_ACTIONS = ALL_ACTIONS
def func_compile_action(action):
    ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
    return ALL_ACTIONS[action]

def func_compile_exp_agent(state, action, rewards, next_state, done):
    global cnt_skip
    global n_skip
    global momentum_opp
    global momentum_ped
    global ema_dist
    global ema_speed
    global last_road_change
    global road_invalid_at_enter

    # Compile reward
    rewards = np.mean(np.array(rewards), axis=0)
    rewards = rewards.tolist()
    rewards.append(np.logical_or(action==1, action==2))  # action == turn?
    print (' '*5+'R: ['+'{:4.2f} '*len(rewards)+']').format(*rewards),

    road_change = rewards[1] > 0.01  # road changed
    road_invalid = rewards[0] > 0.01  # any yellow or red
    if road_change and not last_road_change:
        road_invalid_at_enter = road_invalid
    last_road_change = road_change
    speed = rewards[2]
    obs_risk = rewards[5]

    ema_speed = 0.5*ema_speed + 0.5*speed
    ema_dist = 1.0 if rewards[6] > 2.0 else 0.9 * ema_dist
    momentum_opp = (rewards[3]<0.5)*(momentum_opp+(1-rewards[3]))
    momentum_opp = min(momentum_opp, 20)
    momentum_ped = (rewards[4]>0.5)*(momentum_ped+rewards[4])
    momentum_ped = min(momentum_ped, 12)

    # road_change
    rewards[0] = -100*(
        (road_change and ema_dist>0.2) or (road_change and momentum_ped > 0)
    )*(n_skip-cnt_skip)  # direct penalty
    # velocity
    rewards[2] *= 10
    rewards[2] -= 10
    # opposite
    rewards[3] = -20*(0.9+0.1*momentum_opp)*(momentum_opp>1.0)
    # ped
    rewards[4] = -40*(0.9+0.1*momentum_ped)*(momentum_ped>1.0)
    # obs factor
    rewards[5] *= -100.0
    # dist
    rewards[6] *= 0.0
    # steering
    rewards[-1] *= -40  # -3
    reward = np.sum(rewards)/100.0
    print '{:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}'.format(
        road_invalid_at_enter, momentum_opp, momentum_ped, ema_dist),
    print ': {:5.2f}'.format(reward)

    if road_invalid_at_enter:
        print "[Early stopping] entered invalid road."
        road_invalid_at_enter = False
        # done = True

    if road_change and ema_dist > 0.2:
        print "[Early stopping] turned onto intersection."
        done = True

    if road_change and momentum_ped>0:
        print "[Early stopping] ped onto intersection."
        done = True

    if obs_risk > 1.0:
        print "[Early stopping] hit obstacle."
        done = True

    return state, action, reward, next_state, done

def gen_backend_cmds():
    ws_path = '/Projects/catkin_ws/'
    initialD_path = '/Projects/hobotrl/playground/initialD/'
    backend_path = initialD_path + 'ros_environments/backend_scripts/'
    utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
    backend_cmds = [
        # Parse maps
        ['python', utils_path+'parse_map.py',
         ws_path+'src/Map/src/map_api/data/honda_wider.xodr',
         utils_path+'road_segment_info.txt'],
        # Generate obs and launch file
        ['python', utils_path+'gen_launch_dynamic_v1.py',
         utils_path+'road_segment_info.txt', ws_path,
         utils_path+'honda_dynamic_obs_template.launch',
         32, '--random_n_obs'],
        # start roscore
        ['roscore'],
        # start reward function script
        ['python', backend_path+'gazebo_rl_reward.py'],
        # start road validity node script
        ['python', backend_path+'road_validity.py',
         utils_path+'road_segment_info.txt.signal'],
        # start car_go script
        ['python', backend_path+'car_go.py'],
        # start simulation restarter backend
        ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
    ]
    return backend_cmds


env = DrivingSimulatorEnv(
    address='vmgpu016.hogpu.cc', port='10004',
    backend_cmds=gen_backend_cmds(),
    defs_obs=[
        ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
        ('/decision_result', 'std_msgs.msg.Int16')
    ],
    defs_reward=[
        ('/rl/current_road_validity', 'std_msgs.msg.Int16'),
        ('/rl/entering_intersection', 'std_msgs.msg.Bool'),
        ('/rl/car_velocity', 'std_msgs.msg.Float32'),
        ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
        ('/rl/on_pedestrian', 'std_msgs.msg.Bool'),
        ('/rl/obs_factor', 'std_msgs.msg.Float32'),
        ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
    ],
    defs_action=[('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')],
    rate_action=10.0,
    window_sizes={'obs': 3, 'reward': 3},
    buffer_sizes={'obs': 3, 'reward': 3},
    func_compile_obs=func_compile_obs,
    func_compile_reward=func_compile_reward,
    func_compile_action=func_compile_action,
    step_delay_target=0.5)
# TODO: define these Gym related params insode DrivingSimulatorEnv
env.observation_space = Box(low=0, high=255, shape=(350, 350, 3))
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
env.action_space = Discrete(len(ALL_ACTIONS))
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
            kernel_regularizer=l2_regularizer(scale=1e-2), name='hid2_adv')
        print hid2.shape
        q = layers.dense(inputs=hid2, units=len(AGENT_ACTIONS)*len(TIME_STEP_SCALES), activation=None,
                         kernel_initializer=tf.random_normal_initializer(-3e-3, 3e-3),
                         kernel_regularizer=l2_regularizer(scale=1e-2), name='q')

        print q.shape

    return {"q": q}

lr = tf.get_variable(
    'learning_rate', [], dtype=tf.float32,
    initializer=tf.constant_initializer(1e-3), trainable=False
)
lr_in = tf.placeholder(dtype=tf.float32)
op_set_lr = tf.assign(lr, lr_in)
optimizer_td = tf.train.AdamOptimizer(learning_rate=lr)
target_sync_rate = 1e-3
state_shape = env.observation_space.shape
graph = tf.get_default_graph()
global_step = tf.get_variable(
    'global_step', [], dtype=tf.int32,
    initializer=tf.constant_initializer(0), trainable=False)

# 1 sample ~= 1MB @ 6x skipping
replay_buffer = BigPlayback(
    bucket_cls=BalancedMapPlayback,
    cache_path="./ReplayBufferCache/experimentexperiment",
    capacity=300000, bucket_size=100, ratio_active=0.05, max_sample_epoch=2,
    num_actions=len(AGENT_ACTIONS), upsample_bias=(1,1,1,0.1)
)

gamma = 0.9
_agent = hrl.DQN(
    f_create_q=f_net, state_shape=state_shape,
    # OneStepTD arguments
    num_actions=len(AGENT_ACTIONS), discount_factor=gamma, ddqn=True,
    # target network sync arguments
    target_sync_interval=1,
    target_sync_rate=target_sync_rate,
    # epsilon greeedy arguments
    # greedy_epsilon=0.025,
    # greedy_epsilon=0.05,
    # greedy_epsilon=0.075,
    # greedy_epsilon=0.2,  # 0.2 -> 0.15 -> 0.1
    # greedy_epsilon=CappedLinear(10000, 0.5, 0.05),
    greedy_epsilon=CappedLinear(10000, 0.1, 0.025),
    # optimizer arguments
    network_optimizer=hrl.network.LocalOptimizer(optimizer_td, 1.0),
    # sampler arguments
    sampler=TransitionSampler(replay_buffer, batch_size=8, interval=1, minimum_count=103),
    # checkpoint
    global_step=global_step
 )

def log_info(update_info):
    global action_fraction
    global action_td_loss
    global agent
    global next_state
    global next_action
    global ALL_ACTIONS
    global AGENT_ACTIONS
    global n_agent_steps
    global n_env_steps
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
    global t_step
    global flag_success
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
            if cnt_skip==0:
                print ("{} "+"{:10.5f} "*4+"{}").format(*s)
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
    if cnt_skip == 0 or n_steps == 0:
        next_q_vals_nt = agent._agent.learn_q(np.asarray(next_state)[np.newaxis, :])[0]
        for i, ac in enumerate(AGENT_ACTIONS):
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
            map(lambda x: x[0], AGENT_ACTIONS), next_q_vals_nt))
        max_idx = np.argmax([v for _, v in p_dict])
        p_str = "({:.3f}) ({:3d})[Q_vals]: ".format(
            time.time(), n_steps)
        for i, (a, v) in enumerate(p_dict):
            if a == AGENT_ACTIONS[next_action][0]:
                sym = '|x|' if i==max_idx else ' x '
            else:
                sym = '| |' if i==max_idx else '   '
            p_str += '{}{:3d}: {:8.4f} '.format(sym, a, v)
        print p_str

    cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info \
        and update_info['td_loss'] is not None else 0
    if not done:
        cum_reward += reward

    summary_proto.value.add(tag='t_infer', simple_value=t_infer)
    summary_proto.value.add(tag='t_learn', simple_value=t_learn)
    summary_proto.value.add(tag='t_step', simple_value=t_step)

    if done:
        print ("Episode {} done in {} steps, reward is {}, "
               "average td_loss is {}. No exploration {}").format(
                   n_ep, n_steps, cum_reward,
                   cum_td_loss/n_steps, exploration_off)
        n_env_steps += n_steps
        summary_proto.value.add(tag='n_steps', simple_value=n_steps)
        summary_proto.value.add(tag='n_env_steps', simple_value=n_env_steps)
        summary_proto.value.add(tag='n_agent_steps', simple_value=n_agent_steps)
        summary_proto.value.add(
            tag='num_episode', simple_value=n_ep)
        summary_proto.value.add(
            tag='cum_reward', simple_value=cum_reward)
        summary_proto.value.add(
            tag='per_step_reward', simple_value=cum_reward/n_steps)
        summary_proto.value.add(
            tag='flag_success', simple_value=flag_success)

    return summary_proto

n_interactive = 0
n_skip = 6
update_rate = 6.0
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 0  # num of episode per test run (no exploration)

try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(
        graph=tf.get_default_graph(),
        is_chief=True,
        init_op=tf.global_variables_initializer(),
        logdir='./experiment',
        save_summaries_secs=10,
        save_model_secs=900)

    with sv.managed_session(config=config) as sess, \
         AsynchronousAgent(agent=_agent, method='rate', rate=update_rate) as agent:

        agent.set_session(sess)
        sess.run(op_set_lr, feed_dict={lr_in: 1e-4})
        print "Using learning rate {}".format(sess.run(lr))
        n_env_steps = 0
        n_agent_steps = 0
        action_fraction = np.ones(len(AGENT_ACTIONS), ) / (1.0 * len(AGENT_ACTIONS))
        action_td_loss = np.zeros(len(AGENT_ACTIONS), )
        while True:
            n_ep += 1
            env.env.n_ep = n_ep  # TODO: do this systematically
            exploration_off = (n_ep%n_test==0) if n_test >0 else False
            learning_off = exploration_off
            n_steps = 0
            reward = 0
            skip_reward = 0
            cum_td_loss = 0.0
            cum_reward = 0.0
            done = False

            last_road_change = False
            road_invalid_at_enter = False
            momentum_opp = 0.0
            momentum_ped = 0.0
            ema_speed = 10.0
            ema_dist = 0.0
            update_info = {}
            t_infer, t_step, t_learn = 0, 0, 0

            state  = env.reset()
            proxy_action = agent.act(state, exploration=not exploration_off)
            n_agent_steps += 1
            action = proxy_action / len(TIME_STEP_SCALES)
            n_skip = TIME_STEP_SCALES[proxy_action % len(TIME_STEP_SCALES)]
            cnt_skip = n_skip
            next_state = state
            next_action = action
            # cnt_skip = 1 if next_action == 0 else n_skip
            # cnt_skip = int(n_skip * (1 + np.random.rand()))  # randome start offset to enforce randomness on phase
            log_info(update_info)

            while True:
                n_steps += 1
                cnt_skip -= 1
                update_info = {}
                t_learn, t_infer, t_step = 0, 0, 0

                # Env step
                t = time.time()
                next_state, reward, done, info = env.step(action)
                flag_success = done
                t_step = time.time() - t
                state, action, reward, next_state, done = \
                    func_compile_exp_agent(state, action, reward, next_state, done)
                flag_tail = done
                flag_success = True if flag_success and reward > 0.0 else False
                skip_reward += reward

                if cnt_skip==0 or done:
                    # average rewards during skipping
                    skip_reward /= (n_skip - cnt_skip)
                    # add tail for non-early-stops
                    skip_reward += flag_tail * gamma * skip_reward/ (1-gamma)
                    update_info = agent.step(
                        sess=sess, state=state, action=proxy_action,
                        reward=skip_reward, next_state=next_state,
                        episode_done=done
                    )
                    t = time.time()
                    next_proxy_action = agent.act(next_state, exploration=not exploration_off)
                    next_action = next_proxy_action / len(TIME_STEP_SCALES)
                    n_skip = next_proxy_action % len(TIME_STEP_SCALES)
                    cnt_skip = n_skip
                    n_agent_steps += 1
                    t_infer += time.time() - t
                    skip_reward = 0
                    state, action = next_state, next_action  # s',a' -> s,a
                    action = next_action
                else:
                    action = 3  # no op during skipping

                sv.summary_computed(sess, summary=log_info(update_info))
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
    replay_buffer.close()
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    print "="*30

