#
# -*- coding: utf-8 -*-

import sys

from playground.softq import SoftQLearning, SoftQMPC

sys.path.append(".")

import tensorflow as tf
import numpy as np
import gym

from hobotrl.experiment import ParallelGridSearch, Experiment
import hobotrl.environments as envs
import hobotrl.network as network
import hobotrl.utils as utils


class SoftQExperiment(Experiment):

    def __init__(self, env,
                 f_create_actor,
                 f_create_q,
                 dim_noise,
                 target_sync_interval,
                 target_sync_rate,
                 alpha_exploration=1.0,
                 max_gradient=10.0,
                 m_particle_svgd=16,
                 m_particle_v=16,
                 episode_n=1000,
                 discount_factor=0.99,
                 # sampler arguments
                 update_interval=4,
                 replay_size=100000,
                 batch_size=32,
                 # epsilon greedy arguments
                 network_optimizer_ctor=lambda: network.LocalOptimizer(tf.train.AdamOptimizer(1e-4), grad_clip=10.0)
                 ):
        self._env, \
        self._f_create_actor, \
        self._f_create_q, \
        self._dim_noise, \
        self._target_sync_interval, \
        self._target_sync_rate, \
        self._max_gradient, \
        self._m_particle_svgd, \
        self._m_particle_v, \
        self._episode_n, \
        self._discount_factor, \
        self._update_interval, \
        self._replay_size, \
        self._batch_size, \
        self._network_optimizer_ctor = \
            env, \
            f_create_actor, \
            f_create_q, \
            dim_noise, \
            target_sync_interval, \
            target_sync_rate, \
            max_gradient, \
            m_particle_svgd, \
            m_particle_v, \
            episode_n, \
            discount_factor, \
            update_interval, \
            replay_size, \
            batch_size, \
            network_optimizer_ctor
        self._alpha_exploration = alpha_exploration
        super(SoftQExperiment, self).__init__()

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        action_shape = list(self._env.action_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )

        agent = SoftQLearning(
            f_create_actor=self._f_create_actor,
            f_create_q=self._f_create_q,
            state_shape=state_shape,
            num_actions=action_shape[0],
            dim_noise=self._dim_noise,
            discount_factor=self._discount_factor,
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            alpha_exploration=self._alpha_exploration,
            max_gradient=self._max_gradient,
            update_interval=self._update_interval,
            m_particle_svgd=self._m_particle_svgd,
            m_particle_v=self._m_particle_v,
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            network_optimizer=self._network_optimizer_ctor(),
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=args.render_once,
            )
            runner.episode(self._episode_n)


class SoftQMPCExperiment(Experiment):

    def __init__(self, env,
                 f_create_actor,
                 f_create_q,
                 f_model,
                 dim_noise,
                 target_sync_interval,
                 target_sync_rate,
                 greedy_epsilon=0.2,
                 sample_n=4,
                 horizon_n=4,
                 alpha_exploration=1.0,
                 max_gradient=10.0,
                 m_particle_svgd=16,
                 m_particle_v=16,
                 episode_n=1000,
                 discount_factor=0.99,
                 # sampler arguments
                 update_interval=4,
                 replay_size=100000,
                 batch_size=32,
                 # epsilon greedy arguments
                 network_optimizer_ctor=lambda: network.LocalOptimizer(tf.train.AdamOptimizer(1e-4), grad_clip=10.0)
                 ):
        self._env, \
        self._f_create_actor, \
        self._f_create_q, \
        self._dim_noise, \
        self._target_sync_interval, \
        self._target_sync_rate, \
        self._max_gradient, \
        self._m_particle_svgd, \
        self._m_particle_v, \
        self._episode_n, \
        self._discount_factor, \
        self._update_interval, \
        self._replay_size, \
        self._batch_size, \
        self._network_optimizer_ctor = \
            env, \
            f_create_actor, \
            f_create_q, \
            dim_noise, \
            target_sync_interval, \
            target_sync_rate, \
            max_gradient, \
            m_particle_svgd, \
            m_particle_v, \
            episode_n, \
            discount_factor, \
            update_interval, \
            replay_size, \
            batch_size, \
            network_optimizer_ctor
        self._alpha_exploration = alpha_exploration
        self._greedy_epsilon, self._sample_n, self._horizon_n = greedy_epsilon, sample_n, horizon_n
        self._f_model = f_model
        super(SoftQMPCExperiment, self).__init__()

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)
        action_shape = list(self._env.action_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

        agent = SoftQMPC(
            f_create_actor=self._f_create_actor,
            f_create_q=self._f_create_q,
            f_model=self._f_model,
            state_shape=state_shape,
            num_actions=action_shape[0],
            dim_noise=self._dim_noise,
            discount_factor=self._discount_factor,
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            greedy_epsilon=self._greedy_epsilon,
            sample_n=self._sample_n,
            horizon_n=self._horizon_n,
            max_gradient=self._max_gradient,
            update_interval=self._update_interval,
            m_particle_svgd=self._m_particle_svgd,
            m_particle_v=self._m_particle_v,
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            network_optimizer=self._network_optimizer_ctor(),
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=args.render_once,
            )
            runner.episode(self._episode_n)


class SoftQPendulum(SoftQExperiment):

    def __init__(self, env=None, f_create_actor=None, f_create_q=None, dim_noise=2,
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 alpha_exploration=utils.CappedExp(1e5, 0.2, 0.08),
                 max_gradient=10.0, m_particle_svgd=16, m_particle_v=16, episode_n=1000, discount_factor=0.9,
                 update_interval=4, replay_size=100000, batch_size=32,
                 network_optimizer_ctor=lambda: network.LocalOptimizer(tf.train.AdamOptimizer(1e-4), grad_clip=10.0)):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]
        l2 = 1e-8
        if f_create_actor is None:
            def f(inputs):
                x = tf.concat(inputs, axis=1)
                action = network.Utils.layer_fcs(x, [256, 256], dim_action,
                                                 activation_out=tf.nn.tanh, l2=l2, var_scope="actor")
                return {"action": action}
            f_create_actor = f
        if f_create_q is None:
            def f(inputs):
                x = tf.concat(inputs, axis=1)
                q = network.Utils.layer_fcs(x, [256, 256], 1,
                                            activation_out=None, l2=l2, var_scope="q")
                q = tf.squeeze(q, axis=1)
                return {"q": q}
            f_create_q = f

        super(SoftQPendulum, self).__init__(env, f_create_actor, f_create_q, dim_noise, target_sync_interval,
                                            target_sync_rate, alpha_exploration, max_gradient, m_particle_svgd, m_particle_v, episode_n,
                                            discount_factor, update_interval, replay_size, batch_size,
                                            network_optimizer_ctor)
Experiment.register(SoftQPendulum, "soft q for pendulum")


class SoftQMPCPendulum(SoftQMPCExperiment):

    def __init__(self, env=None, f_create_actor=None, f_create_q=None, f_model=None, dim_noise=2,
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 greedy_epsilon=utils.CappedExp(1e5, 2.5, 0.05),
                 sample_n=4,
                 horizon_n=4,
                 alpha_exploration=0.1,
                 max_gradient=10.0,
                 m_particle_svgd=16,
                 m_particle_v=16,
                 episode_n=1000,
                 discount_factor=0.99,
                 update_interval=4,
                 replay_size=100000,
                 batch_size=32,
                 network_optimizer_ctor=lambda: network.LocalOptimizer(tf.train.AdamOptimizer(1e-4), grad_clip=10.0)):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]
        l2 = 1e-8
        if f_create_actor is None:
            def f(inputs):
                x = tf.concat(inputs, axis=1)
                action = network.Utils.layer_fcs(x, [256, 256], dim_action,
                                                 activation_out=tf.nn.tanh, l2=l2, var_scope="actor")
                return {"action": action}
            f_create_actor = f
        if f_create_q is None:
            def f(inputs):
                x = tf.concat(inputs, axis=1)
                q = network.Utils.layer_fcs(x, [256, 256], 1,
                                            activation_out=None, l2=l2, var_scope="q")
                q = tf.squeeze(q, axis=1)
                return {"q": q}
            f_create_q = f
        if f_model is None:
            def f(inputs):
                state, action = inputs[0], inputs[1]
                se = tf.concat([state, action], axis=-1)
                se = network.Utils.layer_fcs(se, [256], 256, activation_out=None, l2=l2, var_scope="se")
                goal = network.Utils.layer_fcs(se, [], dim_state, activation_out=None, l2=l2, var_scope="goal")
                reward = network.Utils.layer_fcs(se, [], 1, activation_out=None, l2=l2, var_scope="reward")
                reward = tf.squeeze(reward, axis=1)
                return {"goal": goal, "reward": reward}
            f_model = f

        super(SoftQMPCPendulum, self).__init__(env, f_create_actor, f_create_q, f_model, dim_noise,
                                               target_sync_interval, target_sync_rate, greedy_epsilon, sample_n,
                                               horizon_n, alpha_exploration, max_gradient, m_particle_svgd,
                                               m_particle_v, episode_n, discount_factor, update_interval, replay_size,
                                               batch_size, network_optimizer_ctor)
Experiment.register(SoftQMPCPendulum, "sql mpc for pendulum")


if __name__ == '__main__':
    Experiment.main()
