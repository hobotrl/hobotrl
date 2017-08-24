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
# from ros_environments import DrivingSimulatorEnv
from ros_environments import DrivingSimulatorEnvClient as DrivingSimulatorEnv
# ROS
import rospy
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage
# Gym
from gym.spaces import Discrete, Box

# Environment
def compile_reward(rewards):
    return rewards

def compile_reward_agent(rewards):
    global momentum_ped
    global momentum_opp
    rewards = np.mean(np.array(rewards), axis=0)
    print (' '*80+'R: ['+'{:4.2f} '*len(rewards)+']').format(*rewards),

    # obstacle
    rewards[0] *= -100.0
    # distance to
    rewards[1] *= -1.0*(rewards[1]>2.0)
    # velocity
    rewards[2] *= 10
    # opposite
    momentum_opp = (rewards[3]<0.5)*(momentum_opp+(1-rewards[3]))
    momentum_opp = min(momentum_opp, 20)
    rewards[3] = -20*(0.9+0.1*momentum_opp)*(momentum_opp>1.0)
    # ped
    momentum_ped = (rewards[4]>0.5)*(momentum_ped+rewards[4])
    momentum_ped = min(momentum_ped, 12)
    rewards[4] = -40*(0.9+0.1*momentum_ped)*(momentum_ped>1.0)

    reward = np.sum(rewards)/100.0
    print '{:6.4f}, {:6.4f}'.format(momentum_opp, momentum_ped),
    print ': {:7.4f}'.format(reward)
    return reward

def compile_obs(obss):
    obs1 = obss[-1][0]
    return obs1

env = DrivingSimulatorEnv(
    address="localhost", port='22230',
    defs_obs=[('/training/image/compressed', CompressedImage)],
    func_compile_obs=compile_obs,
    defs_reward=[
        ('/rl/has_obstacle_nearby', Bool),
        ('/rl/distance_to_longestpath', Float32),
        ('/rl/car_velocity', Float32),
        ('/rl/last_on_opposite_path', Int16),
        ('/rl/on_pedestrian', Bool)],
    func_compile_reward=compile_reward,
    defs_action=[('/autoDrive_KeyboardMode', Char)],
    rate_action=10.0,
    window_sizes={'obs': 2, 'reward': 3},
    buffer_sizes={'obs': 2, 'reward': 3},
    step_delay_target=0.5)
# TODO: define these Gym related params insode DrivingSimulatorEnv
env.observation_space = Box(low=0, high=255, shape=(640, 640, 3))
env.action_space = Discrete(3)
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
env = FrameStack(env, 3)
ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'd', 'a']]

# Agent
def f_net(inputs):
    inputs = inputs[0]
    inputs = inputs/128 - 1.0
    # (640, 640, 3*n) -> ()
    with tf.device('/gpu:0'):
        conv1 = layers.conv2d(
            inputs=inputs, filters=16, kernel_size=(8, 8), strides=1,
            kernel_regularizer=l2_regularizer(scale=1e-2), name='conv1')
        print conv1.shape
        pool1 = layers.max_pooling2d(
            inputs=conv1, pool_size=3, strides=4, name='pool1')
        print pool1.shape
        conv2 = layers.conv2d(
            inputs=pool1, filters=16, kernel_size=(5, 5), strides=1,
            kernel_regularizer=l2_regularizer(scale=1e-2), name='conv2')
        print conv2.shape
        pool2 = layers.max_pooling2d(
            inputs=conv2, pool_size=3, strides=3, name='pool2')
        print pool2.shape
        conv3 = layers.conv2d(
             inputs=pool2, filters=64, kernel_size=(3, 3), strides=1,
             kernel_regularizer=l2_regularizer(scale=1e-2), name='conv3')
        print conv3.shape
        pool3 = layers.max_pooling2d(
            inputs=conv3, pool_size=3, strides=8, name='pool3',)
        print pool3.shape
        conv4 = layers.conv2d(
            inputs=pool3, filters=64, kernel_size=(3, 3), strides=1,
            kernel_regularizer=l2_regularizer(scale=1e-2), name='conv4')
        print conv4.shape
        pool4 = layers.max_pooling2d(
            inputs=conv4, pool_size=3, strides=8, name='pool4')
        print pool4.shape
        depth = pool4.get_shape()[1:].num_elements()
        inputs = tf.reshape(pool4, shape=[-1, depth])
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
        num_actions=len(ACTIONS), capacity=5000),
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
            #if cnt_skip==n_skip:
            #    print ("{} "+"{:8.5f} "*4+"{}").format(*s)
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
                tag='next_q_vals_{}'.format(ac[0].data),
                simple_value=next_q_vals_nt[i])
            summary_proto.value.add(
                tag='next_q_vals_{}'.format(ac[0].data),
                simple_value=next_q_vals_nt[i])
            summary_proto.value.add(
                tag='action_td_loss_{}'.format(ac[0].data),
                simple_value=action_td_loss[i])
            summary_proto.value.add(
                tag='action_fraction_{}'.format(ac[0].data),
                simple_value=action_fraction[i])
        p_dict = sorted(zip(
            map(lambda x: x[0].data, ACTIONS), next_q_vals_nt))
        max_idx = np.argmax([v for _, v in p_dict])
        p_str = "({:.3f}) ({:3d})[Q_vals]: ".format(
            time.time(), n_steps)
        for i, (a, v) in enumerate(p_dict):
            if a == ACTIONS[next_action][0].data:
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
        else:
            summary_proto.value.add(
                tag='num_episode_noexpolore', simple_value=n_ep)
            summary_proto.value.add(
                tag='cum_reward_noexpolore',
                simple_value=cum_reward)

    return summary_proto

n_interactive = 0
n_skip = 5
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
        action_fraction = np.ones(3,) / (1.0*len(ACTIONS))
        action_td_loss = np.zeros(3,)
        momentum_opp = 0.0
        momentum_ped = 0.0
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
            state, action  = env.reset(), 0
            while True:
                n_steps += 1
                # Env step
                next_state, reward, done, info = env.step(ACTIONS[action])
                reward = compile_reward_agent(reward)
                skip_reward += reward
                done = (reward < -0.9) or done  # heuristic early stopping
                # agent step
                cnt_skip -= 1
                update_info = {}
                if cnt_skip==0 or done:
                    skip_reward /= (n_skip - cnt_skip)
                    if not learning_off:
                        update_info = agent.step(
                            sess=sess, state=state, action=action,
                            reward=skip_reward, next_state=next_state,
                            episode_done=done)
                    next_action = agent.act(next_state, exploration=not exploration_off)
                    cnt_skip = n_skip
                    skip_reward = 0
                    state, action = next_state, next_action  # s',a' -> s,a
                else:
                    if not learning_off:
                        update_info = agent.reinforce_(
                            sess=sess, state=None, action=None,
                            reward=None, next_state=None,
                            episode_done=None)
                    next_action = action
                sv.summary_computed(sess, summary=log_info(update_info))
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

