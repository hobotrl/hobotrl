import sys

import gym
import tensorflow as tf
import numpy as np

from hobotrl.environments import RewardLongerEnv
from hobotrl.experiment import Experiment
from playground.dynamic_gae import OnDQN, OnDPG
import hobotrl as hrl
from exp_atari import full_wrap_dqn, f_dqn_atari
from car import wrap_car


class OnDPGExperiment(Experiment):

    def __init__(self, env,
                 f_se, f_actor, f_critic,
                 episode_n=1000,
                 # ACUpdate arguments
                 discount_factor=0.9,
                 # optimizer arguments
                 network_optimizer_ctor=lambda:hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 # policy arguments
                 ou_params=(0, 0.2, 0.2),
                 # target network sync arguments
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 # sampler arguments
                 batch_size=8,
                 replay_capacity=1000,
                 generation_decay=0.95, neighbour_size=8,
                 **kwargs
                 ):
        self._env, self._f_se, self._f_actor, self._f_critic, self._episode_n,\
            self._discount_factor, self._network_optimizer_ctor, \
            self._ou_params, self._target_sync_interval, self._target_sync_rate, \
            self._batch_size, self._replay_capacity = \
            env, f_se, f_actor, f_critic, episode_n, \
            discount_factor, network_optimizer_ctor, \
            ou_params, target_sync_interval, target_sync_rate, \
            batch_size, replay_capacity
        self._generation_decay, self._neighbour_size, self._kwargs = generation_decay, neighbour_size, kwargs
        super(OnDPGExperiment, self).__init__()

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)
        dim_action = self._env.action_space.shape[-1]
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = OnDPG(
            f_se=self._f_se,
            f_actor=self._f_actor,
            f_critic=self._f_critic,
            state_shape=state_shape,
            dim_action=dim_action,
            # ACUpdate arguments
            discount_factor=self._discount_factor,
            target_estimator=None,
            # optimizer arguments
            network_optimizer=self._network_optimizer_ctor(),
            # policy arguments
            ou_params=self._ou_params,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            batch_size=self._batch_size,
            global_step=global_step,
            generation_decay=self._generation_decay,
            neighbour_size=self._neighbour_size,
            **self._kwargs
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint, render_interval=args.render_interval,
                render_once=True,
                logdir=args.logdir
            )
            runner.episode(self._episode_n)


class OnDQNExperiment(Experiment):

    def __init__(self, env,
                 f_create_q,
                 episode_n=1000,
                 discount_factor=0.99,
                 ddqn=False,
                 # target network sync arguments
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 # sampler arguments
                 update_interval=4,
                 replay_size=1000,
                 batch_size=8,
                 neighbour_size=8,
                 # epsilon greedy arguments
                 greedy_epsilon=0.3,
                 generation_decay=0.95,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 **kwargs
                 ):
        self._env, self._f_create_q, self._episode_n, \
            self._discount_factor, \
            self._ddqn, \
            self._target_sync_interval, \
            self._target_sync_rate, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._generation_decay, \
            self._greedy_epsilon, \
            self._network_optimizer_ctor, \
            self._neighbour_size = \
            env, f_create_q, episode_n, \
            discount_factor, \
            ddqn, \
            target_sync_interval, \
            target_sync_rate, \
            update_interval, \
            replay_size, \
            batch_size, \
            generation_decay, \
            greedy_epsilon, \
            network_optimizer_ctor, \
            neighbour_size
        self._kwargs = kwargs

        super(OnDQNExperiment, self).__init__()

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = OnDQN(
            f_create_q=self._f_create_q,
            neighbour_size=self._neighbour_size,
            state_shape=state_shape,
            # OneStepTD arguments
            num_actions=self._env.action_space.n,
            discount_factor=self._discount_factor,
            ddqn=self._ddqn,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            update_interval=self._update_interval,
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            generation_decay=self._generation_decay,
            # epsilon greedy arguments
            greedy_epsilon=self._greedy_epsilon,
            network_optmizer=self._network_optimizer_ctor(),
            global_step=global_step,
            **self._kwargs
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=sys.maxint, logdir=args.logdir,
                render_once=True,
            )
            return runner.episode(self._episode_n)


class OnDQNPendulum(OnDQNExperiment):
    def __init__(self, env=None, f_create_q=None, episode_n=1000, discount_factor=0.9, ddqn=False, target_sync_interval=100,
                 target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=8, neighbour_size=8,
                 greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.1, 0.02),
                 generation_decay=0.95,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), **kwargs):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.C2DEnvWrapper(env, [5])
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        if f_create_q is None:
            l2 = 1e-8
            activation = tf.nn.elu
            dim_action = env.action_space.n
            def f(inputs):
                input_se = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_se, [256, 256], dim_action,
                                                  activation, None, l2=l2)
                return {"q": out}
            f_create_q = f
        super(OnDQNPendulum, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, update_interval, replay_size, batch_size, neighbour_size,
                                            greedy_epsilon, generation_decay, network_optimizer_ctor, **kwargs)

Experiment.register(OnDQNPendulum, "OnDQNPendulum")


class OnDQNBreakout(OnDQNExperiment):

    def __init__(self, env=None, f_create_q=None, episode_n=10000, discount_factor=0.99, ddqn=False, target_sync_interval=100,
                 target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=8, neighbour_size=8,
                 greedy_epsilon=hrl.utils.CappedLinear(2e5, 0.1, 0.01),
                 generation_decay=0.95,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0), **kwargs):
        if env is None:
            env = gym.make("BreakoutNoFrameskip-v4")
            env = full_wrap_dqn(env)
            env = RewardLongerEnv(env)
        if f_create_q is None:
            f_create_q = f_dqn_atari(env.action_space.n)
        super(OnDQNBreakout, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, update_interval, replay_size, batch_size, neighbour_size,
                                            greedy_epsilon, generation_decay, network_optimizer_ctor, **kwargs)
Experiment.register(OnDQNBreakout, "OnDQNBreakout")


class OnDQNCarRacing(OnDQNBreakout):

    def __init__(self, env=None, f_create_q=None, episode_n=1000, discount_factor=0.99, ddqn=False,
                 target_sync_interval=100, target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=8,
                 neighbour_size=8,
                 greedy_epsilon=hrl.utils.CappedLinear(2e5, 0.1, 0.01),
                 generation_decay=0.95,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0), **kwargs):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = wrap_car(env, 3, 3)
        super(OnDQNCarRacing, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                             target_sync_rate, update_interval, replay_size, batch_size, neighbour_size,
                                             greedy_epsilon, generation_decay, network_optimizer_ctor, **kwargs)
Experiment.register(OnDQNCarRacing, "OnDQNCarRacing")


class OnDPGPendulum(OnDPGExperiment):

    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, episode_n=1000, discount_factor=0.9,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0),
                 ou_params=(0, 0.2, hrl.utils.CappedLinear(1e5, 0.1, 0.01)),
                 target_sync_interval=10, target_sync_rate=0.01, batch_size=8, replay_capacity=1000,
                 generation_decay=0.95, neighbour_size=8, **kwargs):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)

        state_shape = list(env.observation_space.shape)
        dim_action = env.action_space.shape[-1]
        l2 = 1e-8
        if f_se is None:
            def f(inputs):
                return {"se": inputs[0]}
            f_se = f
        if f_actor is None:
            def f(inputs):
                se = inputs[0]
                actor = hrl.network.Utils.layer_fcs(se, [200, 100], dim_action, activation_out=tf.nn.tanh, l2=l2,
                                                    var_scope="action")
                return {"action": actor}
            f_actor = f
        if f_critic is None:
            def f(inputs):
                se, action = inputs[0], inputs[1]
                se = tf.concat([se, action], axis=-1)
                q = hrl.network.Utils.layer_fcs(se, [100], 1, activation_out=None, l2=l2, var_scope="q")
                q = tf.squeeze(q, axis=1)
                return {"q": q}
            f_critic = f

        super(OnDPGPendulum, self).__init__(env, f_se, f_actor, f_critic, episode_n, discount_factor,
                                            network_optimizer_ctor, ou_params, target_sync_interval, target_sync_rate,
                                            batch_size, replay_capacity, generation_decay, neighbour_size, **kwargs)

Experiment.register(OnDPGPendulum, "DPG for Pendulum")


if __name__ == '__main__':
    Experiment.main()