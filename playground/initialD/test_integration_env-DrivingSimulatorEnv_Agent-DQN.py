import sys
sys.path.append('../../')


import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.playback import MapPlayback
from hobotrl.algorithms.dqn import DQN

from ros_environments import DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Float32
from sensor_msgs.msg import Image

# Environment
env = DrivingSimulatorEnv(
    [('/training/image', Image)],
    [('/rl/has_obstacle_nearby', Bool),
     ('/rl/distance_to_longestpath', Float32)],
    [('/autoDrive_KeyboardMode', Char)],
    rate_action=1.0,
    buffer_sizes={'observation': 10, 'reward': 10, 'action': 10}
)
ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'd', 'a']]

# Agent
def f_net(inputs, num_outputs, is_training):
    inputs = inputs/128 - 1.0
    conv1 = layers.conv2d(
        inputs=inputs, filters=16, kernel_size=(8, 8), strides=1,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='conv1'
    )
    pool1 = layers.max_pooling2d(
        inputs=conv1, pool_size=3, strides=16, name='pool1'
    )
    conv2 = layers.conv2d(
        inputs=pool1, filters=16, kernel_size=(5, 5), strides=1,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='conv2'
    )
    pool2 = layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=64, name='pool2'
    )
    # conv3 = layers.conv2d(
    #     inputs=pool2, filters=64, kernel_size=(3, 3), strides=1,
    #     kernel_regularizer=l2_regularizer(scale=1e-2), name='conv3'
    #)
    #pool3 = layers.max_pooling2d(
    #    inputs=conv3, pool_size=3, strides=8, name='pool3',
    #)
    #conv4 = layers.conv2d(
    #    inputs=pool3, filters=64, kernel_size=(3, 3), strides=1,
    #    kernel_regularizer=l2_regularizer(scale=1e-2), name='conv4'
    #)
    #pool4 = layers.max_pooling2d(
    #    inputs=conv3, pool_size=3, strides=8, name='pool4'
    #)
    depth = pool2.get_shape()[1:].num_elements()
    inputs = tf.reshape(pool2, shape=[-1, depth])
    hid1 = layers.dense(
        inputs=inputs, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='hid1'
    )
    hid2 = layers.dense(
        inputs=hid1, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='hid2'
    )
    q = layers.dense(
        inputs=hid2, units=num_outputs, activation=None,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='q'
    )
    q = tf.squeeze(q, name='out_sqz')
    return q

optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
target_sync_rate = 0.01
training_params = (optimizer_td, target_sync_rate, 10.0)
state_shape = (640, 640, 3)
graph = tf.get_default_graph()

agent = DQN(
    # EpsilonGreedyPolicyMixin params
    actions=range(len(ACTIONS)),
    epsilon=0.2,
    # DeepQFuncMixin params
    dqn_param_dict={
        'gamma': 0.9,
        'f_net': f_net,
        'state_shape': state_shape,
        'num_actions':len(ACTIONS),
        'training_params':training_params,
        'schedule':(1, 10),
        'greedy_policy':True,
        'ddqn': False,
        'graph':graph
    },
    # ReplayMixin params
    buffer_class=MapPlayback,
    buffer_param_dict={
        "capacity": 1000,
        "sample_shapes": {
            'state': state_shape,
            'action': (),
            'reward': (),
            'next_state': state_shape,
            'episode_done': ()
         }},
    batch_size=8
)

n_interactive = 0

try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    agent.set_session(sess)
    while True:
        cum_reward = 0.0
        n_steps = 0
        cum_td_loss = 0.0
        state, action = env.reset(), np.random.randint(0, len(ACTIONS))
        next_state, reward, done, info = env.step(ACTIONS[action])
        while True:
            n_steps += 1
            cum_reward += reward
            next_action, update_info = agent.step(
                sess=sess,
                state=map(lambda x: (x-2)/5.0, state),  # scale state to [-1, 1]
                action=action,
                reward=float(reward>1.0),  # reward clipping
                next_state=map(lambda x: (x-2)/5.0, next_state), # scle state
                episode_done=done,
            )
            cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info is not None else 0
            # print update_info
            if done is True:
                print "Episode done in {} steps, reward is {}, average td_loss is {}".format(
                    n_steps, cum_reward, cum_td_loss/n_steps
                )
                n_steps = 0
                cum_reward = 0.0
                if n_interactive == 0:
                    raw_input('Next episode?')
                    n_interactive = 100
                else:
                    n_interactive -= 1
                break
            state, action = next_state, next_action
            next_state, reward, done, info = env.step(ACTIONS[action])
except rospy.ROSInterruptException:
    pass
finally:
    sess.close()
