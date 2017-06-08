#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import gym
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer


import hobotrl as hrl
from hobotrl.experiment import Experiment
import hobotrl.algorithms.ac as ac
import hobotrl.algorithms.dqn as dqn


class ACPendulum(Experiment):
    def run(self, args):
        env = gym.make("Pendulum-v0")
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1)

        def create_value_net(inputs, num_outputs, is_training):
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
                activation=None,
                kernel_regularizer=l2_regularizer(scale=1e-4),
                trainable=True, name='out',
            )
            q = tf.squeeze(q, name='out_sqz')
            return q

        def create_policy_net(inputs, num_action):
            input_var = inputs[0]
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return tf.nn.softmax(fc_out, name="softmax")

        state_shape = list(env.observation_space.shape)
        agent = ac.ActorCritic(
            state_shape=state_shape,
            num_actions=env.action_space.n,
            f_create_policy=create_policy_net,
            f_create_value=create_value_net,
            entropy=0.01,
            gamma=0.9,
            train_interval=8,
            batch_size=8,
            training_params=(tf.train.AdamOptimizer(learning_rate=0.0001), 0.01),
            schedule=(8, 4),
            buffer_param_dict={
                "capacity": 16,
                "sample_shapes": {
                    'state': state_shape,
                    'action': [1],
                    'reward': [1],
                    'next_state': state_shape,
                    'episode_done': [1]
                 }
            },
            # EpsilonGreedyPolicyMixin params
            epsilon=0.02
        )
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        agent.set_session(sess)
        runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
        runner.episode(1000)

Experiment.register(ACPendulum, "discrete actor critic for Pendulum")


class DQNPendulum(Experiment):
    def run(self, args):
        env = gym.make("Pendulum-v0")
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1)

        optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        target_sync_rate = 0.01
        training_params = (optimizer_td, target_sync_rate)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return fc_out

        state_shape = list(env.observation_space.shape)
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # DeepQFuncMixin params
            gamma=0.9,
            f_net=f_net, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=training_params, schedule=(1, 10),
            greedy_policy=True,
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
            buffer_param_dict={
                "capacity": 1000,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (1,),
                    'reward': (1,),
                    'next_state': state_shape,
                    'episode_done': (1,)
                }},
            batch_size=8
        )
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        agent.set_session(sess)
        runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
        runner.episode(1000)

Experiment.register(DQNPendulum, "DQN for Pendulum")


class DDQNPendulum(Experiment):
    def run(self, args):
        env = gym.make("Pendulum-v0")
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1)

        optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        target_sync_rate = 0.01
        training_params = (optimizer_td, target_sync_rate)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return fc_out

        state_shape = list(env.observation_space.shape)
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # DeepQFuncMixin params
            gamma=0.9,
            f_net=f_net, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=training_params, schedule=(1, 10),
            greedy_policy=True,
            ddqn=True,
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
            buffer_param_dict={
                "capacity": 1000,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (1,),
                    'reward': (1,),
                    'next_state': state_shape,
                    'episode_done': (1,)
                }},
            batch_size=8
        )
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        agent.set_session(sess)
        runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
        runner.episode(1000)

Experiment.register(DDQNPendulum, "Double DQN for Pendulum")


if __name__ == '__main__':
    Experiment.main()