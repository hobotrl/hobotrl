# -*- coding: utf-8 -*-

"""Intergration test for stochastic policy gradient (mixin-based)

TODO: wrap up with Experiement?
"""

import sys
sys.path.append('../../../')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.playback import MapPlayback

from hobotrl.algorithms.ac import ActorCritic
from hobotrl.environments import GridworldSink

# Environment
env = GridworldSink()

# Agent
def f_net_value(inputs, num_outputs, is_training):
    depth = inputs.get_shape()[1:].num_elements()
    inputs = tf.reshape(inputs, shape=[-1, depth])
    hidden1 = layers.dense(
        inputs=inputs, units=200,
        activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='hidden1',
    )
    hidden2 = layers.dense(
        inputs=hidden1, units=200,
        activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='hidden2',
    )
    q = layers.dense(
        inputs=hidden2, units=num_outputs,
        activation=tf.nn.tanh,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='out',
    )
    q = tf.squeeze(q, name='out_sqz')
    return q

def f_net_policy(inputs, num_outputs):
    inputs = inputs[0]
    depth = inputs.get_shape()[1:].num_elements()
    inputs = tf.reshape(inputs, shape=[-1, depth])
    hidden1 = layers.dense(
        inputs=inputs, units=200,
        activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='hidden1',
    )
    hidden2 = layers.dense(
        inputs=hidden1, units=200,
        activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='hidden2',
    )
    action_dist = layers.dense(
        inputs=hidden2, units=num_outputs,
        activation=tf.nn.softmax,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='out',
    )
    return action_dist

gamma = 0.9
optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer_pg = tf.train.GradientDescentOptimizer(learning_rate=0.01)
target_sync_rate = 0.01
training_params_td = (optimizer_td, target_sync_rate, 10.0)
training_params_pg = (optimizer_pg,)
state_shape = (len(env.DIMS),)
graph = tf.get_default_graph()

agent = ActorCritic(
    # DeepStochasticPolicyMixin
    dsp_param_dict={
        'state_shape': state_shape,
        'num_actions': len(env.ACTIONS),
        'is_continuous_action': False,
        'f_create_net': f_net_policy,
        'training_params': training_params_pg,
        'entropy': 0.001
    },
    backup_method='multistep',
    update_interval=3,
    gamma=gamma,
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
    batch_size=8,
    # DeepQFuncMixin params
    dqn_param_dict={
        'gamma': gamma,
        'f_net': f_net_value,
        'state_shape': state_shape,
        'num_actions':len(env.ACTIONS),
        'training_params':training_params_td,
        'schedule':(1, 10),
        'greedy_policy':False,
        'ddqn': False,
        'graph':graph
    },
)

n_interactive = 0
cum_steps = 0.0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
agent.set_session(sess)
while True:
    cum_reward = 0.0
    n_steps = 0
    cum_td_loss = 0.0
    cum_spg_loss = 0.0
    state, action = env.reset(), np.random.randint(0, len(env.ACTIONS))
    next_state, reward, done, info = env.step(env.ACTIONS[action])
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
        cum_spg_loss += update_info['spg_loss'] if 'spg_loss' in update_info is not None else 0
        # print update_info
        if done is True:
            print ("Episode done in {} steps, reward is {}, "
                   "average td_loss is {}, "
                   "average spg_loss is {}"
                  ).format(
                n_steps, cum_reward,
                cum_td_loss/n_steps, cum_spg_loss/n_steps
            )
            # print agent._DeepStochasticPolicyMixin__dsp.distribution.dist_run(
            #     agent.sess, [np.array([[-0.4, -0.4]])]
            #)
            cum_steps += n_steps
            n_steps = 0
            cum_reward = 0.0
            if n_interactive == 0:
                print "total {} steps".format(cum_steps)
                cum_steps = 0.0
                raw_input('Next episode?')
                n_interactive = 100
            else:
                n_interactive -= 1
            break
        state, action = next_state, next_action
        next_state, reward, done, info = env.step(env.ACTIONS[action])
sess.close()

