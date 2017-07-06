# -*- coding: utf-8 -*-

"""Intergration test for deep deterministic policy (mixin-based)
"""

import sys
sys.path.append('../../../')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.playback import MapPlayback
from hobotrl.algorithms.dpg import DPG

import gym

# Environment
def video_callable(episode_id):
    return episode_id%50 == 0
env = gym.make('Pendulum-v0')
env = gym.wrappers.Monitor(
    env, './test_integration_env-Pendulum-v0_agent_DPG/test02/',
    force=True,
    # video_callable=video_callable
)
# Agent
def f_net_dqn(inputs_state, inputs_action, is_training):
    depth = inputs_state.get_shape()[1:].num_elements()
    inputs_state = tf.reshape(inputs_state, shape=[-1, depth])
    inputs_state = layers.batch_normalization(inputs_state, axis=1, training=is_training)
    hidden1 = layers.dense(
        inputs=inputs_state, units=400,
        activation=None,
        kernel_regularizer=l2_regularizer(scale=1e-2),
        trainable=True, name='hidden1',
    )
    # hidden1 = tf.nn.relu(hidden1)
    hidden1 = tf.nn.relu(layers.batch_normalization(hidden1, axis=1, training=is_training))
    depth = inputs_action.get_shape()[1:].num_elements()
    inputs_action = tf.reshape(inputs_action, shape=[-1, depth])
    hidden1 = tf.concat([hidden1, inputs_action], axis=1)
    hidden2 = layers.dense(
        inputs=hidden1, units=300,
        activation=None,
        kernel_regularizer=l2_regularizer(scale=1e-2),
        trainable=True, name='hidden2',
    )
    hidden2 = tf.nn.relu(hidden2)
    # hidden2 = tf.nn.relu(layers.batch_normalization(hidden2, axis=1, training=is_training))
    q = layers.dense(
        inputs=hidden2, units=1,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
        kernel_regularizer=l2_regularizer(scale=1e-2),
        trainable=True, name='out',
    )
    q = tf.squeeze(q, axis=1, name='out_sqz')
    return q


def f_net_ddp(inputs, action_shape, is_training):
    depth_state = inputs.get_shape()[1:].num_elements()
    inputs = tf.reshape(inputs, shape=[-1, depth_state], name='inputs')
    inputs = layers.batch_normalization(inputs, axis=1, training=is_training)
    hidden1 = layers.dense(
        inputs=inputs, units=400,
        activation=None,
        kernel_regularizer=l2_regularizer(scale=1e-2),
        trainable=True, name='hidden1',
    )
    # hidden1 = tf.nn.relu(hidden1)
    hidden1 = tf.nn.relu(layers.batch_normalization(hidden1, axis=1, training=is_training))
    hidden2 = layers.dense(
        inputs=hidden1, units=300,
        activation=None,
        kernel_regularizer=l2_regularizer(scale=1e-2),
        trainable=True, name='hidden2',
    )
    # hidden2 = tf.nn.relu(layers.batch_normalization(hidden2, axis=1, training=is_training))
    hidden2 = tf.nn.relu(hidden2)
    depth_action = reduce(lambda x, y: x*y, action_shape, 1)
    action = layers.dense(
        inputs=hidden2, units=depth_action,
        activation=tf.nn.tanh,
        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
        kernel_regularizer=l2_regularizer(scale=1e-2),
        trainable=True, name='out'
    )
    action = tf.reshape(2.0*action, shape=[-1]+list(action_shape), name='out')

    return action

ou_params = (0.0, 0.15, 0.2)
optimizer_td = tf.train.AdamOptimizer(learning_rate=1e-3)
optimizer_dpg = tf.train.AdamOptimizer(learning_rate=1e-4)
target_sync_rate = 1e-3
training_params_dqn = (optimizer_td, target_sync_rate, 10.0)
training_params_ddp = (optimizer_dpg, target_sync_rate, 10.0)
schedule_ddp = (1, 1)
schedule_dqn = (1, 1)
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
batch_size = 64
graph = tf.get_default_graph()

agent = DPG(
    # === ReplayMixin params ===
    buffer_class=MapPlayback,
    buffer_param_dict={
        "capacity": 1000000,
        "sample_shapes": {
            'state': state_shape,
            'action': action_shape,
            'reward': (),
            'next_state': state_shape,
            'episode_done': ()
         }},
    batch_size=batch_size,
    # === OUExplorationMixin ===
    ou_params=ou_params,
    action_shape=action_shape,
    # === DeepDeterministicPolicyMixin ===
    ddp_param_dict={
        'f_net': f_net_ddp,
        'state_shape': state_shape,
        'action_shape': action_shape,
        'training_params': training_params_ddp,
        'schedule': schedule_ddp,
        'graph': graph
    },
    # === DeepQFuncMixin params ===
    dqn_param_dict={
        'gamma': 0.99,
        'f_net': f_net_dqn,
        'state_shape': state_shape,
        'action_shape': action_shape,
        'training_params': training_params_dqn,
        'schedule': schedule_dqn,
        'greedy_policy': True,
        'graph': graph
    },
    is_action_in=True
)

N_interactive = 0
n_interactive = N_interactive - 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())
agent.set_session(sess)
while True:
    cum_reward = 0.0
    n_steps = 0
    cum_td_loss = 0.0
    cum_q_vals = 0.0
    cum_nq_vals = 0.0
    cum_dqn_diff = 0.0
    cum_ddp_diff = 0.0
    cum_grad_norm = 0.0
    cum_grad_norm_td = 0.0
    state, action = env.reset(), env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    while True:
        n_steps += 1
        cum_reward += reward
        next_action, update_info = agent.step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            episode_done=done,
        )
        # print n_steps,
        # print (state, action, reward)
        # print update_info['td_loss'] if 'td_loss' in update_info else None,
        # q_vals = update_info['q_vals'] if 'q_vals' in update_info else 0
        # td_tar = update_info['td_target'] if 'td_target' in update_info else 0
        # print update_info['dqn_target_diff_l2'] if 'dqn_target_diff_l2' in update_info else None,
        # print update_info['ddp_target_diff_l2'] if 'ddp_target_diff_l2' in update_info else None,
        # print np.mean((q_vals - td_tar)**2),
        # print
        cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info else 0
        cum_q_vals += np.mean(update_info['q_vals']) if 'q_vals' in update_info else 0
        cum_nq_vals += np.mean(update_info['next_q_vals']) if 'next_q_vals' in update_info else 0
        cum_grad_norm += update_info['dpg_grad_norm'] if 'dpg_grad_norm' in update_info else 0
        cum_grad_norm_td += update_info['td_grad_norm'] if 'td_grad_norm' in update_info else 0
        cum_dqn_diff += update_info['dqn_target_diff_l2'] if 'dqn_target_diff_l2' in update_info else 0
        cum_ddp_diff += update_info['ddp_target_diff_l2'] if 'ddp_target_diff_l2' in update_info else 0

        if done is True:
            print ("Episode done in {} steps, reward {:.6f}, q {:.6f}, next_q {:.6f}, "
            "td_loss {:.6f}, dqn_diff {:.6f}, ddp_diff {:.6f}, dpg_norm {:.6f}, td_norm {:.6f}").format(
                n_steps, cum_reward/n_steps, cum_q_vals/n_steps,
                cum_nq_vals/n_steps, cum_td_loss/n_steps,
                cum_dqn_diff/n_steps, cum_ddp_diff/n_steps,
                cum_grad_norm/n_steps, cum_grad_norm_td/n_steps
            )
            if n_interactive == 0:
                raw_input('Next episode?')
                n_interactive = N_interactive - 1
            else:
                n_interactive -= 1 if n_interactive>0 else 0
            break
        state, action = next_state, next_action
        next_state, reward, done, info = env.step(action)
sess.close()

