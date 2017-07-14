#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import logging

import cv2
import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

import hobotrl as hrl
from hobotrl.utils import LinearSequence
from hobotrl.experiment import Experiment
import hobotrl.algorithms.ac as ac
import hobotrl.algorithms.dqn as dqn
import hobotrl.algorithms.per as per
import playground.optimal_tighten as play_ot
import hobotrl.algorithms.ot as ot


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
                                      dtype=tf.int32,
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
                                      dtype=tf.int32,
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
                                      dtype=tf.int32,
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
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=sys.maxint, render_interval=sys.maxint, logdir=args.logdir)
            runner.episode(500)

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
                                      dtype=tf.int32,
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
                                      dtype=tf.int32,
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
                                      dtype=tf.int32,
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
        n_episodes = 500

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return fc_out

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
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
                "priority_bias": 0.5,  # todo search what combination of exponent/importance_correction works better
                "importance_weight": LinearSequence(n_episodes * 200, 0.5, 1.0),

        },
            batch_size=8,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, evaluate_interval=100, render_interval=50, logdir=args.logdir)
            runner.episode(n_episodes)

Experiment.register(PERDQNPendulum, "Prioritized Exp Replay with DQN, for Pendulum")


class OTDQNPendulum(Experiment):
    """
    converges on Pendulum.
    However, in Pendulum, weight_upper > 0 hurts performance.
    should verify on more difficult problems
    """
    def run(self, args):
        reward_decay = 0.9
        K = 4
        batch_size = 8
        weight_lower = 1.0
        weight_upper = 1.0
        target_sync_interval = 10
        replay_size = 1000

        env = gym.make("Pendulum-v0")
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.1)

        optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        target_sync_rate = 0.01
        training_params = (optimizer_td, target_sync_rate)

        def f_net(inputs, num_action):
            input_var = inputs
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return fc_out

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = play_ot.OTDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # OTDQN
            f_net=f_net,
            state_shape=state_shape,
            action_n=env.action_space.n,
            reward_decay=reward_decay,
            batch_size=batch_size,
            K=K,
            weight_lower=weight_lower,
            weight_upper=weight_upper,
            optimizer=optimizer_td,
            target_sync_interval=target_sync_interval,
            replay_capacity=replay_size,
            # BaseDeepAgent
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=sys.maxint, logdir=args.logdir)
            runner.episode(500)

Experiment.register(OTDQNPendulum, "Optimaly Tightening DQN for Pendulum")


class AOTDQNPendulum(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    should verify on more difficult problems
    """
    def run(self, args):
        reward_decay = 0.9
        K = 4
        batch_size = 8
        weight_lower = 1.0
        weight_upper = 1.0
        replay_size = 1000

        env = gym.make("Pendulum-v0")
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.1)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return fc_out

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = ot.OTDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # OTDQN
            f_net_dqn=f_net,
            state_shape=state_shape,
            num_actions=env.action_space.n,
            reward_decay=reward_decay,
            batch_size=batch_size,
            K=K,
            weight_lower_bound=weight_lower,
            weight_upper_bound=weight_upper,
            training_params=(tf.train.AdamOptimizer(learning_rate=0.001), 0.01),
            schedule=(1, 10),
            replay_capacity=replay_size,
            # BaseDeepAgent
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=sys.maxint, logdir=args.logdir)
            runner.episode(500)

Experiment.register(AOTDQNPendulum, "Optimaly Tightening DQN for Pendulum")


class AOTDQNBreakout(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    should verify on more difficult problems
    """
    def run(self, args):
        reward_decay = 0.9
        K = 4
        batch_size = 8
        weight_lower = 1.0
        weight_upper = 1.0
        replay_size = 1000

        env = gym.make("Breakout-v0")
        # env = hrl.envs.C2DEnvWrapper(env, [5])

        def state_trans(state):
            gray = np.asarray(np.dot(state, [0.299, 0.587, 0.114]))
            gray = cv2.resize(gray, (84, 84))
            return np.asarray(gray.reshape(gray.shape + (1,)), dtype=np.uint8)

        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.1,
                                         state_augment_proc=state_trans, state_stack_n=4)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            print "input size:", input_var
            out = hrl.utils.Network.conv2d(input_var=input_var, h=8, w=8, out_channel=32,
                                           strides=[4, 4], activation=tf.nn.relu, var_scope="conv1")
            # 20 * 20 * 32
            print "out size:", out
            out = hrl.utils.Network.conv2d(input_var=out, h=4, w=4, out_channel=64,
                                           strides=[2, 2], activation=tf.nn.relu, var_scope="conv2")
            # 9 * 9 * 64
            print "out size:", out
            out = hrl.utils.Network.conv2d(input_var=out, h=3, w=3, out_channel=64,
                                           strides=[1, 1], activation=tf.nn.relu, var_scope="conv3")
            # 7 * 7 * 64
            print "out size:", out
            out = tf.reshape(out, [-1, 7 * 7 * 64])
            out = hrl.utils.Network.layer_fcs(input_var=out, shape=[512], out_count=num_action,
                                              activation_hidden=tf.nn.relu,
                                              activation_out=None, var_scope="fc")
            return out

        state_shape = [84, 84, 4]  # list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        agent = ot.OTDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # OTDQN
            f_net_dqn=f_net,
            state_shape=state_shape,
            num_actions=env.action_space.n,
            reward_decay=reward_decay,
            batch_size=batch_size,
            K=K,
            weight_lower_bound=weight_lower,
            weight_upper_bound=weight_upper,
            training_params=(tf.train.AdamOptimizer(learning_rate=0.001), 0.01),
            schedule=(1, 10),
            replay_capacity=replay_size,
            state_offset_scale=(-128, 1.0 / 128),
            # BaseDeepAgent
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=sys.maxint, logdir=args.logdir)
            runner.episode(500)

Experiment.register(AOTDQNBreakout, "Optimaly Tightening DQN for Breakout")


class BootstrappedDQNSnakeGame(Experiment):
    def run(self, args):
        """
        Run the experiment.
        """
        def render():
            """
            Render the environment and related information to the console.
            """
            if not display:
                return

            print env.render(mode='ansi')
            print "Reward:", reward
            print "Head:", agent.current_head
            print "Done:", done
            print ""
            time.sleep(frame_time)

        from environments.snake import SnakeGame
        from hobotrl.algorithms.bootstrapped_DQN import BootstrappedDQN
        from hobotrl.environments import EnvRunner2

        import time
        import os
        import random

        # Parameters
        random.seed(1105)  # Seed

        for n_head in [15, 20]:

            log_dir = os.path.join(args.logdir, "head%d" % n_head)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file_name = "booststrapped_DQN_Snake.csv"

            # Initialize the environment and the agent
            env = SnakeGame(3, 3, 1, 1, max_episode_length=30)
            agent = BootstrappedDQN(observation_space=env.observation_space,
                                    action_space=env.action_space,
                                    reward_decay=1.,
                                    td_learning_rate=0.5,
                                    target_sync_interval=2000,
                                    nn_constructor=self.nn_constructor,
                                    loss_function=self.loss_function,
                                    trainer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize,
                                    replay_buffer_class=hrl.playback.MapPlayback,
                                    replay_buffer_args={"capacity": 20000},
                                    min_buffer_size=100,
                                    batch_size=20,
                                    n_heads=n_head)

            # Start training
            env_runner = EnvRunner2(env=env,
                                    agent=agent,
                                    n_episodes=3000,
                                    moving_average_window_size=100,
                                    no_reward_reset_interval=-1,
                                    checkpoint_save_interval=1000,
                                    log_dir=log_dir,
                                    log_file_name=log_file_name,
                                    render_env=False,
                                    render_interval=1000,
                                    render_length=200,
                                    frame_time=0.1,
                                    render_options={"mode": "ansi"}
                                    )
            env_runner.run()
            # env_runner.run_demo("17000.ckpt")

    @staticmethod
    def loss_function(output, target):
        """
        Calculate the loss.
        """
        return tf.reduce_sum(tf.sqrt(tf.squared_difference(output, target)+1)-1, axis=-1)

    @staticmethod
    def nn_constructor(observation_space, action_space, n_heads, **kwargs):
        """
        Construct the neural network.
        """
        def leakyRelu(x):
            return tf.maximum(0.01*x, x)

        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

        def weight(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        def bias(shape):
            return tf.Variable(tf.constant(0.1, shape=shape))

        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        eshape = observation_space.shape
        nn_outputs = []

        # Layer 1 parameters
        n_channel1 = 8
        w1 = weight([3, 3, eshape[-1], n_channel1])
        b1 = bias([n_channel1])

        # Layer 2 parameters
        n_channel2 = 16
        w2 = weight([n_channel1*eshape[0]*eshape[1], n_channel2])
        b2 = bias([n_channel2])

        # Layer 1
        layer1 = leakyRelu(conv2d(x, w1) + b1)
        layer1_flatten = tf.reshape(layer1, [-1, n_channel1*eshape[0]*eshape[1]])

        # Layer 2
        layer2 = leakyRelu(tf.matmul(layer1_flatten, w2) + b2)

        for i in range(n_heads):
            # Layer 3 parameters
            w3 = weight([n_channel2, 4])
            b3 = bias([4])

            # Layer 3
            layer3 = tf.matmul(layer2, w3) + b3

            nn_outputs.append(layer3)

        return {"input": x, "head": nn_outputs}

Experiment.register(BootstrappedDQNSnakeGame, "Bootstrapped DQN for the Snake game")


class BootstrappedDQNCartPole(Experiment):
    def run(self, args):
        """
        Run the experiment.
        """
        from hobotrl.algorithms.bootstrapped_DQN import BootstrappedDQN
        from hobotrl.environments import EnvRunner2

        import os

        n_head = 10  # Number of heads

        log_dir = args.logdir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = "bootstrapped_DQN_Pendulum.csv"

        # Initialize the environment and the agent
        env = gym.make('CartPole-v0')
        # env = hrl.envs.C2DEnvWrapper(env, [5])
        agent = BootstrappedDQN(observation_space=env.observation_space,
                                action_space=env.action_space,
                                reward_decay=1.,
                                td_learning_rate=0.5,
                                target_sync_interval=2000,
                                nn_constructor=self.nn_constructor,
                                loss_function=self.loss_function,
                                trainer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize,
                                replay_buffer_class=hrl.playback.MapPlayback,
                                replay_buffer_args={"capacity": 10000},
                                min_buffer_size=1000,
                                batch_size=10,
                                n_heads=n_head)

        env_runner = EnvRunner2(env=env,
                                agent=agent,
                                n_episodes=-1,
                                moving_average_window_size=50,
                                no_reward_reset_interval=-1,
                                checkpoint_save_interval=2000,
                                log_dir=log_dir,
                                log_file_name=log_file_name,
                                render_env=True,
                                render_interval=4000,
                                render_length=200,
                                frame_time=0.1
                                )
        env_runner.run()

    @staticmethod
    def loss_function(output, target):
        """
        Calculate the loss.
        """
        return tf.reduce_sum(tf.squared_difference(output, target), axis=-1)

    @staticmethod
    def nn_constructor(observation_space, action_space, n_heads, **kwargs):
        """
        Construct the neural network.
        """
        def weight(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        def bias(shape):
            return tf.Variable(tf.constant(0.1, shape=shape))

        eshape = observation_space.shape[0]
        nn_outputs = []

        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        # Layer 1 parameters
        n_channel1 = 16
        w1 = weight([eshape, n_channel1])
        b1 = bias([n_channel1])

        # Layer 2 parameters
        n_channel2 = 8
        w2 = weight([n_channel1, n_channel2])
        b2 = bias([n_channel2])

        # Layer 1
        layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

        # Layer 2
        layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

        for i in range(n_heads):
            # Layer 3 parameters
            w3 = weight([n_channel2, action_space.n])
            b3 = bias([action_space.n])

            # Layer 3
            layer3 = tf.matmul(layer2, w3) + b3

            nn_outputs.append(layer3)

        return {"input": x, "head": nn_outputs}

Experiment.register(BootstrappedDQNCartPole, "Bootstrapped DQN for the CartPole")

from hobotrl.algorithms.bootstrapped_DQN import BootstrappedDQN


class BootstrappedDQNAtari(Experiment):
    def __init__(self, env, augment_wrapper_args={}, agent_args={}, runner_args={},
                 stack_n=4, frame_skip_n=4, reward_decay=0.99,
                 agent_type=BootstrappedDQN):
        """
        Base class Experiments in Atari games.

        :param env: environment.
        :param augment_wrapper_args(dict): arguments for "AugmentEnvWrapper".
        :param agent_args(dict): arguments for the agent.
        :param runner_args(dict): arguments for the environment runner.
        :param agent_type(class): class name of the agent.
        """
        assert stack_n >= 1
        assert 1 <= frame_skip_n <= stack_n
        assert stack_n % frame_skip_n == 0
        assert 0. <= reward_decay <= 1.

        import math

        Experiment.__init__(self)

        n_head = 10  # Number of heads

        self.augment_wrapper_args = augment_wrapper_args
        self.agent_args = agent_args
        self.runner_args = runner_args

        # Wrap the environment
        history_stack_n = stack_n//frame_skip_n
        augment_wrapper_args = {"reward_decay": math.pow(reward_decay, 1.0/history_stack_n),
                                "reward_scale": 1.,
                                "state_augment_proc": self.state_trans,
                                "state_stack_n": frame_skip_n,
                                "state_scale": 1.0/255.0}
        augment_wrapper_args.update(self.augment_wrapper_args)
        env = self.env = hrl.envs.AugmentEnvWrapper(env, **augment_wrapper_args)
        env = self.env = hrl.envs.StateHistoryStackEnvWrapper(env,
                                                              reward_decay=math.pow(reward_decay, 1.0/history_stack_n),
                                                              stack_n=history_stack_n)

        # Initialize the agent
        agent_args = {"reward_decay": reward_decay,
                      "td_learning_rate": 1.,
                      "target_sync_interval": 1000,
                      "nn_constructor": self.nn_constructor,
                      "loss_function": self.loss_function,
                      "trainer": tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize,
                      "replay_buffer_class": hrl.playback.MapPlayback,
                      "replay_buffer_args": {"capacity": 25000},
                      "min_buffer_size": 10000,
                      "batch_size": 8,
                      "n_heads": n_head}
        agent_args.update(self.agent_args)
        self.agent = agent_type(observation_space=env.observation_space,
                                action_space=env.action_space,
                                **agent_args)

    @staticmethod
    def state_trans(state):
        """
        Transform the state to 84*84 grayscale image.

        :param state: state.
        :return: transformed image.
        """
        gray = np.asarray(np.dot(state, [0.299, 0.587, 0.114]))
        gray = cv2.resize(gray, (84, 84))

        return np.asarray(gray.reshape(gray.shape + (1,)), dtype=np.int8)

    @staticmethod
    def show_state_trans_result_wrapper(state):
        """
        Transform the state with "state_trans" and show the result in the image viewer.

        :param state: state.
        :return: transformed image
        """
        global image_viewer
        import gym.envs.classic_control.rendering as rendering

        # Initialize image viewer if needed
        try:
            image_viewer
        except NameError:
            image_viewer = rendering.SimpleImageViewer()

        # Transform with state_trans
        image = BootstrappedDQNAtari.state_trans(state)

        # Resize the image to see it clearly
        im_view = image.reshape((84, 84))
        im_view = np.array(im_view, dtype=np.float32)
        im_view = cv2.resize(im_view, (336, 336), interpolation=cv2.INTER_NEAREST)
        im_view = np.array(im_view, dtype=np.int8)
        im_view = np.stack([im_view]*3, axis=-1)

        # Show image
        image_viewer.imshow(im_view)
        return image

    def run(self, args, checkpoint_number=None):
        """
        Run the experiment.

        :param args: arguments.
        :param checkpoint_number: if not None, checkpoint will be loaded before training.
        """
        from hobotrl.environments import EnvRunner2
        import os

        # Create logging folder if needed
        log_dir = args.logdir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = "booststrapped_DQN.csv"

        # Initialize the environment runner
        runner_args = {"n_episodes": -1,
                       "moving_average_window_size": 100,
                       "no_reward_reset_interval": -1,
                       "checkpoint_save_interval": 100000,
                       "render_env": False,
                       "show_frame_rate": True,
                       "show_frame_rate_interval": 2000}
        runner_args.update(self.runner_args)
        env_runner = EnvRunner2(env=self.env,
                                agent=self.agent,
                                log_dir=log_dir,
                                log_file_name=log_file_name,
                                **runner_args)

        # Load checkpoint if needed
        if checkpoint_number:
            checkpoint_file_name = '%d.ckpt' % checkpoint_number
            env_runner.load_checkpoint(checkpoint_file_name, checkpoint_number)

        # Start training
        env_runner.run()

    @staticmethod
    def loss_function(output, target):
        """
        Calculate the loss.
        """
        return tf.reduce_sum(tf.sqrt(tf.squared_difference(output, target)+1)-1, -1)

    @staticmethod
    def nn_constructor(observation_space, action_space, n_heads, **kwargs):
        """
        Construct the neural network.
        """
        def leakyRelu(x):
            return tf.maximum(0.01*x, x)

        import tensorflow.contrib.layers as layers
        nn_outputs = []

        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        print "input size:", x
        out = hrl.utils.Network.conv2d(input_var=x, h=8, w=8, out_channel=32,
                                       strides=[4, 4], activation=leakyRelu, var_scope="conv1")
        # 20 * 20 * 32
        print "out size:", out
        out = hrl.utils.Network.conv2d(input_var=out, h=4, w=4, out_channel=64,
                                       strides=[2, 2], activation=leakyRelu, var_scope="conv2")
        # 9 * 9 * 64
        print "out size:", out
        out = hrl.utils.Network.conv2d(input_var=out, h=3, w=3, out_channel=64,
                                       strides=[1, 1], activation=leakyRelu, var_scope="conv3")

        # 7 * 7 * 64
        out = tf.reshape(out, [-1, int(np.product(out.shape[1:]))])
        out = layers.fully_connected(out, 512, activation_fn=leakyRelu)
        print "out size:", out

        for _ in range(n_heads):
            head = layers.fully_connected(out, action_space.n, activation_fn=None)

            nn_outputs.append(head)

        return {"input": x, "head": nn_outputs}


class BootstrappedDQNBattleZone(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('BattleZone-v0'),
                                      augment_wrapper_args={"reward_scale": 0.001},
                                      agent_args={"replay_buffer_args": {"capacity": 10000},
                                                  "min_buffer_size": 10000})

    def run(self, args, **kwargs):
        BootstrappedDQNAtari.run(self, args, **kwargs)

Experiment.register(BootstrappedDQNBattleZone, "Bootstrapped DQN for the BattleZone")


class BootstrappedDQNBreakOut(BootstrappedDQNAtari):
    def __init__(self):
        from hobotrl.algorithms.bootstrapped_DQN import bernoulli_mask
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('Breakout-v0'),
                                      runner_args={"no_reward_reset_interval": 2000},
                                      agent_args={"n_heads": 30,
                                                  "bootstrap_mask": bernoulli_mask(0.2)}
                                      )

Experiment.register(BootstrappedDQNBreakOut, "Bootstrapped DQN for the BreakOut")


class BootstrappedDQNPong(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self, gym.make('PongNoFrameskip-v4'))

Experiment.register(BootstrappedDQNPong, "Bootstrapped DQN for the Pong")


class BootstrappedDQNEnduro(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('Enduro-v0'),
                                      augment_wrapper_args={
                                          "reward_scale": 0.3
                                          })

    def run(self, args, **kwargs):
        BootstrappedDQNAtari.run(self, args, checkpoint_number=1300000)

Experiment.register(BootstrappedDQNEnduro, "Bootstrapped DQN for the Enduro")


class BootstrappedDQNIceHockey(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('IceHockey-v0'),
                                      augment_wrapper_args={
                                          "reward_scale": 1.0
                                          },
                                      agent_args={
                                          "batch_size": 3,
                                      },
                                      # runner_args={"render_env": True,
                                      #              "frame_time": 0.05}
                                      frame_skip_n=1
                                      )

    def run(self, args, **kwargs):
        BootstrappedDQNAtari.run(self, args, checkpoint_number=2000000)

Experiment.register(BootstrappedDQNIceHockey, "Bootstrapped DQN for the IceHockey")


class BootstrappedDQNKangaroo(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('Kangaroo-v0'),
                                      augment_wrapper_args={
                                          "reward_scale": 1.0
                                          },
                                      runner_args={"render_env": True,
                                                   "frame_time": 0.05}
                                      )

Experiment.register(BootstrappedDQNKangaroo, "Bootstrapped DQN for the Kangaroo")


class RandomizedBootstrappedDQNBreakOut(BootstrappedDQNAtari):
    def __init__(self):
        import math
        from hobotrl.algorithms.bootstrapped_DQN import RandomizedBootstrappedDQN

        def eps_function(step):
            return 0.1*(math.cos(step/4.0e5*math.pi) + 1)

        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('Breakout-v0'),
                                      runner_args={"no_reward_reset_interval": 2000,
                                                   # "render_env": True,
                                                   # "frame_time": 0.05
                                                   },
                                      agent_args={"eps_function": eps_function},  # LinearSequence(1e6, 0.2, 0.0)},
                                      agent_type=RandomizedBootstrappedDQN
                                      )

Experiment.register(RandomizedBootstrappedDQNBreakOut, "Randomized Bootstrapped DQN for the Breakout")


def demo_experiment_generator(experiment_class, checkpoint_file_name, frame_time=0.05):
    """
    Generate a demo experiment using "EnvRunner2".

    :param experiment_class: class of the experiment.
    :param checkpoint_file_name: file name of the checkpoint that should be loaded.
    :param frame_time: will be passed to the environment runner.
    :return: an experiment.
    """
    class BootstrappedDQNDemo(Experiment):
        def run(self, args):
            from hobotrl.environments import EnvRunner2

            experiment = experiment_class()
            env_runner = EnvRunner2(env=experiment.env,
                                    agent=experiment.agent,
                                    log_dir=args.logdir,
                                    frame_time=frame_time)
            env_runner.run_demo(checkpoint_file_name)

    BootstrappedDQNDemo.__name__ = experiment_class.__name__ + "Demo"
    return BootstrappedDQNDemo

Experiment.register(demo_experiment_generator(RandomizedBootstrappedDQNBreakOut, "3300000.ckpt"), "Demo for the Breakout")
Experiment.register(demo_experiment_generator(BootstrappedDQNPong, "1080000.ckpt"), "Demo for the Pong")
Experiment.register(demo_experiment_generator(BootstrappedDQNBattleZone, "2232000.ckpt"), "Demo for the Battle Zone")
Experiment.register(demo_experiment_generator(BootstrappedDQNEnduro, "1300000.ckpt", frame_time=0.0), "Demo for the Enduro")
Experiment.register(demo_experiment_generator(BootstrappedDQNIceHockey, "1900000.ckpt", frame_time=0.01), "Demo for the Ice Hockey")


if __name__ == '__main__':
    Experiment.main()
