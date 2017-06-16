#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import logging

import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer


import hobotrl as hrl
from hobotrl.experiment import Experiment
import hobotrl.algorithms.ac as ac
import hobotrl.algorithms.dqn as dqn
import hobotrl.algorithms.per as per


class ACDiscretePendulum(Experiment):
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
            # q = tf.squeeze(q, name='out_sqz')
            return q

        def create_policy_net(inputs, num_action):
            input_var = inputs[0]
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return tf.nn.softmax(fc_out, name="softmax")

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = ac.ActorCritic(
            state_shape=state_shape,
            is_continuous_action=False,
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
                    'action': [],
                    'reward': [],
                    'next_state': state_shape,
                    'episode_done': []
                 }
            },
            # EpsilonGreedyPolicyMixin params
            epsilon=0.02,
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
            runner.episode(1000)


Experiment.register(ACDiscretePendulum, "discrete actor critic for Pendulum")


class ACContinuousPendulum(Experiment):
    def run(self, args):
        env = gym.make("Pendulum-v0")
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1,
                                         action_limit=np.asarray([env.action_space.low, env.action_space.high]))

        def create_value_net(state, action, is_training):
            l2 = 1e-4
            status_encoder = hrl.utils.Network.layer_fcs(state, [200], 200,
                                                         activation_hidden=tf.nn.relu, activation_out=tf.nn.relu,
                                                         l2=l2, var_scope="se")
            input_var = tf.concat([status_encoder, action], axis=-1)
            q = hrl.utils.Network.layer_fcs(input_var,[100], 1,
                                            activation_hidden=tf.nn.relu, activation_out=None,
                                            l2=l2, var_scope="q")
            q = tf.reshape(q, [-1])
            return q

        def create_policy_net(inputs, num_action):
            l2 = 1e-4
            input_var = inputs[0]
            status_encoder = hrl.utils.Network.layer_fcs(input_var, [200], 200,
                                                         activation_hidden=tf.nn.relu, activation_out=tf.nn.relu,
                                                         l2=l2, var_scope="se")
            mean = hrl.utils.Network.layer_fcs(status_encoder, [], num_action,
                                               activation_out=tf.nn.tanh,
                                               l2=l2, var_scope="mean")
            stddev = hrl.utils.Network.layer_fcs(status_encoder, [], num_action,
                                                 activation_out=tf.nn.softplus,
                                                 l2=l2, var_scope="stddev")
            return {"stddev": stddev, "mean": mean}

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        action_dim = env.action_space.shape[0]
        agent = ac.ActorCritic(
            state_shape=state_shape,
            is_continuous_action=True,
            num_actions=action_dim,
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
                    'action': [action_dim],
                    'reward': [],
                    'next_state': state_shape,
                    'episode_done': []
                 }
            },
            # EpsilonGreedyPolicyMixin params
            epsilon=0.02,
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
            runner.episode(1000)


Experiment.register(ACContinuousPendulum, "continuous actor critic for Pendulum")


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
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
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
                    'action': (),
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                }},
            batch_size=8,
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
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
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # DeepQFuncMixin params
            gamma=0.9,
            f_net_dqn=f_net, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=training_params, schedule=(1, 10),
            greedy_policy=True,
            ddqn=True,
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
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
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
            runner.episode(1000)

Experiment.register(DDQNPendulum, "Double DQN for Pendulum")


class DuelDQNPendulum(Experiment):
    def run(self, args):
        env = gym.make("Pendulum-v0")
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1)

        optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        target_sync_rate = 0.01
        training_params = (optimizer_td, target_sync_rate)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            se = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=tf.nn.relu, l2=1e-4)
            v = hrl.utils.Network.layer_fcs(se, [100], 1, var_scope="v")
            a = hrl.utils.Network.layer_fcs(se, [100], num_action, var_scope="a")
            a = a - tf.reduce_mean(a, axis=1, keep_dims=True)
            q = a + v
            return q

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # DeepQFuncMixin params
            gamma=0.9,
            f_net_dqn=f_net, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=training_params, schedule=(1, 10),
            greedy_policy=True,
            ddqn=False,
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
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
            graph=tf.get_default_graph(),
            global_step=global_step
        )

        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                              init_op=tf.global_variables_initializer(), save_dir=args.logdir)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
            runner.episode(1000)

Experiment.register(DuelDQNPendulum, "Duel DQN for Pendulum")


class CarEnvWrapper(object):
    """
    Wraps car env into discrete action control problem
    """

    def __init__(self, env, steer_n, speed_n):
        self.steer_n, self.speed_n = steer_n, speed_n
        self.env = env
        self.action_n = steer_n * speed_n
        self.action_space = gym.spaces.discrete.Discrete(self.action_n)

    def __getattr__(self, name):
        print("getattr:", name, " @ ", id(self.env))
        if name == "action_space":
            print("getattr: action_space:", name)
            return self.action_space
        else:
            return getattr(self.env, name)

    def step(self, *args, **kwargs):
        # lives_before = self.env.ale.lives()
        if len(args) > 0:
            action_i = args[0]
        else:
            action_i = kwargs["action"]
        action_c = self.action_d2c(action_i)
        # logging.warning("action d2c: %s => %s", action_i, action_c)
        next_state, reward, done, info = self.env.step(action_c)
        # lives_after = self.env.ale.lives()
        #
        # # End the episode when a life is lost
        # if lives_before > lives_after:
        #   done = True
        #
        # # Clip rewards to [-1,1]
        # reward = max(min(reward, 1), -1)

        return next_state, reward, done, info

    def action_c2d(self, action):
        """
        continuous action to discrete action
        :param action:
        :return:
        """
        steer_i = int((action[0] - (-1.0)) / 2.0 * self.steer_n)
        steer_i = self.steer_n - 1 if steer_i >= self.steer_n else steer_i
        if abs(action[1]) > abs(action[2]):
            speed_action = action[1]
        else:
            speed_action = -action[2]
        speed_i = int((speed_action - (-1.0)) / 2.0 * self.speed_n)
        speed_i = self.speed_n - 1 if speed_i >= self.speed_n else speed_i
        return steer_i * self.speed_n + speed_i

    def action_d2c(self, action):
        steer_i = int(action / self.speed_n)
        speed_i = action % self.speed_n
        action_c = np.asarray([0., 0., 0.])
        action_c[0] = float(steer_i) / self.steer_n * 2 - 1.0 + 1.0 / self.steer_n
        speed_c = float(speed_i) / self.speed_n * 2 - 1.0 + 1.0 / self.speed_n
        if speed_c >= 0:
            action_c[1], action_c[2] = speed_c, 0
        else:
            action_c[1], action_c[2] = 0, -speed_c
        return action_c


class DQNCarRacing(Experiment):
    def run(self, args):
        reward_decay = 0.9

        env = gym.make("CarRacing-v0")
        env = CarEnvWrapper(env, 3, 3)
        env = hrl.envs.AugmentEnvWrapper(env,
                                         reward_decay=reward_decay,
                                         # reward_scale=0.1,
                                         state_stack_n=4)

        optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        target_sync_rate = 0.01
        training_params = (optimizer_td, target_sync_rate)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            # 96 * 96 * C
            out = hrl.utils.Network.conv2d(input_var, 8, 8, 32, strides=[4, 4], var_scope="conv1")
            # 23 * 23 * 32
            out = hrl.utils.Network.conv2d(out, 5, 5, 64, strides=[2, 2], var_scope="conv2")
            # 10 * 10 * 64
            out = hrl.utils.Network.conv2d(out, 4, 4, 64, strides=[1, 1], var_scope="conv3")
            # 7 * 7 * 64
            out = tf.reshape(out, [-1, 7 * 7 * 64])
            out = hrl.utils.Network.layer_fcs(out, [], 512, activation_out=tf.nn.relu, var_scope="fc4")
            v = hrl.utils.Network.layer_fcs(out, [100], 1, activation_out=tf.nn.relu, var_scope="v")
            a = hrl.utils.Network.layer_fcs(out, [100], num_action, activation_out=tf.nn.relu, var_scope="a")
            a = a - tf.reduce_mean(a, axis=1, keep_dims=True)
            q = a + v
            return q
            # q = hrl.utils.Network.layer_fcs(out, [], num_action, activation_out=None, var_scope="fc5")
            # return q

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # DeepQFuncMixin params
            gamma=reward_decay,
            f_net_dqn=f_net, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=training_params, schedule=(1, 10),
            greedy_policy=True,
            ddqn=True,
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
            buffer_param_dict={
                "capacity": 10000,
                "sample_shapes": {
                    'state': state_shape,
                    'action': [],
                    'reward': [],
                    'next_state': state_shape,
                    'episode_done': []
                },
                "augment_offset": {
                    'state': -128,
                    'next_state': -128,
                },
                "augment_scale": {
                    'state': 1.0/128,
                    'next_state': 1.0/128,
                }
            },
            batch_size=32,
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
            runner.episode(100000)

Experiment.register(DQNCarRacing, "DQN for CarRacing, tuned with ddqn, duel network, etc.")


class PERDQNPendulum(Experiment):
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
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = per.PrioritizedDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # DeepQFuncMixin params
            gamma=0.9,
            f_net_dqn=f_net, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=training_params, schedule=(1, 10),
            greedy_policy=True,
            # ReplayMixin params
            buffer_class=hrl.playback.NearPrioritizedPlayback,
            buffer_param_dict={
                "capacity": 1000,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (),
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                },
                "exponent": 0.  # todo search what combination of exponent/importance_correction works better
            },
            batch_size=8,
            importance_correction=0.,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
            runner.episode(1000)

Experiment.register(PERDQNPendulum, "Prioritized Exp Replay with DQN, for Pendulum")

if __name__ == '__main__':
    Experiment.main()