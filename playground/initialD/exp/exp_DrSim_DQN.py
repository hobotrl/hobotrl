import os
import signal
import time
import sys
import traceback
sys.path.append('../../..')
sys.path.append('..')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

import hobotrl as hrl

#from hobotrl.playback import MapPlayback
# from hobotrl.algorithms.dqn import DQN
#from dqn import DQNSticky

from ros_environments import DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage

# Environment
def compile_reward(rewards):
    rewards = map(
        lambda reward: sum((
            -100.0 * float(reward[0]),  # obstacle 0 or -0.04
             -1.0 * float(reward[1])*(float(reward[1])>2.0),  # distance to 0.002 ~ 0.008
             10.0 * float(reward[2]),  # car_velo 0 ~ 0.08
            -20.0 * (1 - float(reward[3])),  # opposite 0 or -0.02
            -70.0 * float(reward[4]),  # ped 0 ~ -0.07
        )),
        rewards)
    return np.mean(rewards)/1000.0

def compile_obs(obss):
    obs1 = obss[-1][0]
    obs2 = obss[-3][0]
    obs3 = obss[-5][0]
    obs = np.concatenate([obs1, obs2, obs3], axis=2)
    return obs

env = DrivingSimulatorEnv(
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
    window_sizes={'obs': 5, 'reward': 5},
    buffer_sizes={'obs': 5, 'reward': 5},
    step_delay_target=1.0
)
ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'd', 'a']]


# Agent
def f_net(inputs):
    inputs = inputs[0]
    inputs = inputs/128 - 1.0
    # (640, 640, 3*n) -> ()
    with tf.device('/gpu:1'):
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
        q = tf.squeeze(q, name='out_sqz', axis=1)
    return {"q": q}

optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
target_sync_rate = 0.001
training_params = (optimizer_td, target_sync_rate, 10.0)
state_shape = (640, 640, 3*3)
graph = tf.get_default_graph()
global_step = tf.get_variable(
    'global_step', [], dtype=tf.int32,
    initializer=tf.constant_initializer(0), trainable=False
)

agent = hrl.DQN(
    f_create_q=f_net, state_shape=state_shape,
    # OneStepTD arguments
    num_actions=len(ACTIONS), discount_factor=0.9,
    ddqn=False,
    # target network sync arguments
    target_sync_interval=1,
    target_sync_rate=0.001,
    # epsilon greeedy arguments
    greedy_epsilon=0.2,
    # optimizer arguments
    network_optimizer=hrl.network.LocalOptimizer(tf.train.GradientDescentOptimizer(1e-3), 10.0),
    max_gradient=10.0,
    # sampler arguments
    update_interval=1, replay_size=5000, batch_size=8,
    # sticky mass todo
    global_step=global_step
)

n_interactive = 0
n_ep = 265  # last ep in the last run, if restart use 0
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
        save_model_secs=1800)

    with sv.managed_session(config=config) as sess:
        agent.set_session(sess)
        while True:
            n_ep += 1
            cum_reward = 0.0
            n_steps = 0
            cum_td_loss = 0.0
            exploration_off = (n_ep%n_test==0)
            state = env.reset()
            action = agent.act(state, exploration_off=exploration_off)
            next_state, reward, done, info = env.step(ACTIONS[action])
            while True:
                n_steps += 1
                cum_reward += reward
                if not exploration_off:
                    update_info = agent.step(
                        sess=sess, state=state, action=action,
                        reward=reward, next_state=next_state,
                        episode_done=done,

                        learning_off=exploration_off,
                        exploration_off=exploration_off
                    )
                else:
                    update_info = {}
                next_action = agent.act(next_state, on=not exploration_off)
                cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info \
                    and update_info['td_loss'] is not None else 0
                summary_proto = tf.Summary()
                for tag in update_info:
                    summary_proto.value.add(
                        tag=tag, simple_value=np.mean(update_info[tag]))
                if 'q_vals' in update_info and \
                   update_info['q_vals'] is not None:
                    q_vals = update_info['q_vals']
                    summary_proto.value.add(
                        tag='q_vals_max',
                        simple_value=np.max(q_vals))
                    summary_proto.value.add(
                        tag='q_vals_min',
                        simple_value=np.min(q_vals))
                next_q_vals_nt = agent.get_value(next_state)
                for i, action in enumerate(ACTIONS):
                    summary_proto.value.add(
                        tag='next_q_vals_{}'.format(action[0].data),
                        simple_value=next_q_vals_nt[i])
                p_dict = sorted(zip(
                    map(lambda x: x[0].data, ACTIONS), next_q_vals_nt))
                max_idx = np.argmax([v for _, v in p_dict])
                p_str = "({:.3f}) [Q_vals]: ".format(time.time())
                for i, (a, v) in enumerate(p_dict):
                    p_str += '{}{:3d}: {:.5f}  '.format(
                        '|-|' if i==max_idx else ' '*3,
                        a, v)
                print p_str
                # print update_info
                if done is True:
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
                sv.summary_computed(sess, summary=summary_proto)
                if done is True:
                    break
                state, action = next_state, next_action
                next_state, reward, done, info = env.step(ACTIONS[action])
except Exception as e:
    print e.message
    traceback.print_exc()
finally:
    print "="*30
    print "="*30
    print "Tidying up..."
    # kill orphaned monitor daemon process
    env.exit()
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    print "="*30

