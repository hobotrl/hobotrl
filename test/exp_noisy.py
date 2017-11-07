
import sys
sys.path.append(".")

import numpy as np
import tensorflow as tf
import gym

from playground.noisy import NoisySD
from playground.noisy_dpg import NoisyDPG
from hobotrl.experiment import Experiment
import hobotrl as hrl


class NoisyExperiment(Experiment):

    def __init__(self,
                 env,
                 f_se,  # state encoder
                 f_manager,  # manager outputs Dstate
                 f_explorer,  # noisy network
                 f_ik,  # inverse kinetic
                 f_value,  # value network
                 f_model,  # transition model network
                 f_pi,
                 episode_n=1000,
                 discount_factor=0.9,
                 noise_dimension=2,
                 se_dimension=4,
                 manager_horizon=32,
                 manager_interval=2,
                 batch_size=32,
                 batch_horizon=4,
                 noise_stddev=1.0,
                 noise_explore_param=(0, 0.2, 0.2),
                 worker_explore_param=(0, 0.2, 0.2),
                 worker_entropy=1e-2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 replay_size=1000,
                 ):
        super(NoisyExperiment, self).__init__()
        self._env, self._f_se, self._f_manager, self._f_explorer, self._f_ik, self._f_value, self._f_model, self._f_pi, \
            self._episode_n, self._discount_factor, \
            self._noise_dimension, self._se_dimension, self._manager_horizon, self._manager_interval, self._batch_size, self._batch_horizon, \
            self._noise_stddev, self._noise_explore_param, self._worker_explore_param, \
            self._network_optimizer_ctor, self._replay_size, self._worker_entropy = \
            env, f_se, f_manager, f_explorer, f_ik, f_value, f_model, f_pi, \
            episode_n, discount_factor, \
            noise_dimension, se_dimension, manager_horizon, manager_interval, batch_size, batch_horizon, \
            noise_stddev, noise_explore_param, worker_explore_param, \
            network_optimizer_ctor, replay_size, worker_entropy

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        action_dimension = self._env.action_space.shape[0]
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
        agent = NoisySD(
            f_se=self._f_se,
            f_manager=self._f_manager,
            f_explorer=self._f_explorer,
            f_ik=self._f_ik,
            f_value=self._f_value,
            f_model=self._f_model,
            f_pi=self._f_pi,
            state_shape=state_shape,
            action_dimension=action_dimension,
            noise_dimension=self._noise_dimension,
            se_dimension=self._se_dimension,
            discount_factor=self._discount_factor,
            network_optimizer=self._network_optimizer_ctor(),
            noise_stddev=self._noise_stddev,
            noise_explore_param=self._noise_explore_param,
            worker_explore_param=self._worker_explore_param,
            worker_entropy=self._worker_entropy,
            manager_horizon=self._manager_horizon,
            manager_interval=self._manager_interval,
            batch_size=self._batch_size,
            batch_horizon=self._batch_horizon,
            replay_size=self._replay_size,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=args.render_once
            )
            runner.episode(self._episode_n)


class NoisyPendulum(NoisyExperiment):
    def __init__(self, env=None, se_dimension=4, f_se=None, f_manager=None, f_explorer=None, f_ik=None, f_value=None, f_model=None, f_pi=None,
                 episode_n=2000, discount_factor=0.9,
                 noise_dimension=2, manager_horizon=16, manager_interval=1, batch_size=8, batch_horizon=4,
                 noise_stddev=hrl.utils.CappedLinear(1e5, 0.5, 0.05),
                 # noise_stddev=0.3,
                 noise_explore_param=(0, 0.2, 0.2),
                 # worker_explore_param=(0, 0.2, 0.2),
                 worker_explore_param=(0, 0.2, hrl.utils.CappedLinear(2e5, 0.2, 0.01)),
                 worker_entropy=1e-2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0),
                 replay_size=1000):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)

        action_dimension = env.action_space.shape[0]
        l2 = 1e-5
        activation = tf.nn.elu
        hidden_dim = 32
        if f_se is None:
            def f(inputs):
                input_state = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_state, [], se_dimension,
                                                  None, None, l2=l2)
                return {"se": out}
            f_se = f

        if f_manager is None:
            def f(inputs):
                input_state = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_state, [hidden_dim], se_dimension,
                                                  activation, tf.nn.tanh, l2=l2)
                return {"sd": out}
            f_manager = f

        if f_explorer is None:
            def f(inputs):
                input_se, input_noise = inputs[0], inputs[1]
                input_var = tf.concat([input_se, input_noise], axis=-1)
                out = hrl.network.Utils.layer_fcs(input_var, [hidden_dim, hidden_dim], se_dimension,
                                                  activation, None, l2=l2)
                return {"noise": out}
            f_explorer = f

        if f_ik is None:
            def f(inputs):
                input_se, input_goal = inputs[0], inputs[1]
                input_var = tf.concat([input_se, input_goal], axis=-1)
                out = hrl.network.Utils.layer_fcs(input_var, [hidden_dim, hidden_dim], action_dimension,
                                                  activation, tf.nn.tanh, l2=l2)
                return {"action": out}
            f_ik = f

        if f_value is None:
            def f(inputs):
                input_se = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_se, [hidden_dim, hidden_dim], 1,
                                                  activation, None, l2=l2)
                out = tf.squeeze(out, axis=-1)
                return {"value": out}
            f_value = f

        if f_model is None:
            def f(inputs):
                input_se, input_action = inputs[0], inputs[1]
                input_var = tf.concat([input_se, input_action], axis=-1)
                out = hrl.network.Utils.layer_fcs(input_var, [hidden_dim, hidden_dim], se_dimension,
                                                  activation, None, l2=l2, var_scope="sd")
                r = hrl.network.Utils.layer_fcs(input_var, [hidden_dim, hidden_dim], 1,
                                                  activation, None, l2=l2, var_scope="r")
                r = tf.squeeze(r, axis=-1)
                return {"sd": out, "r": r}
            f_model = f
        if f_pi is None:
            def f(inputs):
                input_se = inputs[0]
                mean = hrl.network.Utils.layer_fcs(input_se, [hidden_dim, hidden_dim], action_dimension,
                                                   activation, tf.nn.tanh, l2=l2, var_scope="mean")
                stddev = hrl.network.Utils.layer_fcs(input_se, [hidden_dim, hidden_dim], action_dimension,
                                                   activation, None, l2=l2, var_scope="stddev")
                stddev = 2.0 * (1.0 + hrl.tf_dependent.ops.atanh(stddev / 4.0))
                return {"mean": mean, "stddev": stddev}
            f_pi = f

        super(NoisyPendulum, self).__init__(env, f_se, f_manager, f_explorer, f_ik, f_value, f_model, f_pi,
                                            episode_n, discount_factor,
                                            noise_dimension, se_dimension, manager_horizon, manager_interval, batch_size, batch_horizon,
                                            noise_stddev, noise_explore_param, worker_explore_param, worker_entropy,
                                            network_optimizer_ctor, replay_size)
Experiment.register(NoisyPendulum, "Noisy explore for pendulum")


class NoisyDPGExperiment(Experiment):

    def __init__(self,
                 # sampler arguments
                 env,
                 f_se,
                 f_actor,
                 f_critic,
                 f_noise,
                 episode_n=1000,
                 discount_factor=0.9,
                 batch_size=32,
                 dim_noise=2,
                 noise_stddev=1.0,
                 noise_weight=1.0,
                 ou_params=(0.0, 0.2, 0.2),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 replay_size=1000,
                 ):
        super(NoisyDPGExperiment, self).__init__()
        self._env, \
            self._f_se, \
            self._f_actor, \
            self._f_critic, \
            self._f_noise, \
            self._episode_n, \
            self._discount_factor, \
            self._batch_size, \
            self._dim_noise, \
            self._noise_stddev, \
            self._noise_weight, \
            self._ou_params, \
            self._network_optimizer_ctor, \
            self._target_sync_interval, \
            self._target_sync_rate, \
            self._replay_size = \
            env, \
            f_se, \
            f_actor, \
            f_critic, \
            f_noise, \
            episode_n, \
            discount_factor, \
            batch_size, \
            dim_noise, \
            noise_stddev, \
            noise_weight, \
            ou_params, \
            network_optimizer_ctor, \
            target_sync_interval, \
            target_sync_rate, \
            replay_size

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        dim_action = self._env.action_space.shape[0]
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
        agent = NoisyDPG(
            self._f_se,
            self._f_actor,
            self._f_critic,
            self._f_noise,
            state_shape, dim_action,
            self._dim_noise,
            # ACUpdate arguments
            self._discount_factor,
            # optimizer arguments
            network_optimizer=self._network_optimizer_ctor(),
            # policy arguments
            ou_params=self._ou_params,
            noise_stddev=self._noise_stddev,
            noise_weight=self._noise_weight,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=args.render_once
            )
            runner.episode(self._episode_n)


class NoisyDPGPendulum(NoisyDPGExperiment):
    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, f_noise=None,
                 episode_n=1000, discount_factor=0.9, batch_size=32,
                 dim_se=16, dim_hidden=32, dim_noise=2,
                 noise_stddev=hrl.utils.CappedLinear(1e5, 0.1, 0.02),
                 noise_weight=1e-1,
                 ou_params=(0.0, 0.2, 0.2),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), target_sync_interval=10,
                 target_sync_rate=0.01, replay_size=1000):

        if env is None:
            env = gym.make("Pendulum-v0")
            # env = hrl.envs.MaxAndSkipEnv(env, 1, skip=2)
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)

        dim_action = env.action_space.shape[0]
        l2 = 1e-5
        activation = tf.nn.elu
        if f_se is None:
            def f(inputs):
                input_state = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_state, [], dim_se,
                                                  None, None, l2=l2)
                return {"se": out}

            f_se = f

        if f_actor is None:
            def f(inputs):
                input_state = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_state, [dim_hidden], dim_action,
                                                  activation,
                                                  # hrl.tf_dependent.ops.atanh,
                                                  tf.nn.tanh,
                                                  l2=l2)
                return {"action": out}

            f_actor = f

        if f_noise is None:
            def f(inputs):
                input_se, input_noise = inputs[0], inputs[1]
                input_var = tf.concat([input_se, input_noise], axis=-1)
                out = hrl.network.Utils.layer_fcs(input_var, [dim_hidden, dim_hidden], dim_action,
                                                  activation,
                                                  # hrl.tf_dependent.ops.atanh,
                                                  tf.nn.tanh,
                                                  l2=l2)
                return {"noise": out}

            f_noise = f

        if f_critic is None:
            def f(inputs):
                input_se, input_action = inputs[0], inputs[1]
                input_var = tf.concat([input_se, input_action], axis=-1)
                out = hrl.network.Utils.layer_fcs(input_var, [dim_hidden, dim_hidden], 1,
                                                  activation, None, l2=l2)
                out = tf.squeeze(out, axis=-1)
                return {"q": out}

            f_critic = f

        super(NoisyDPGPendulum, self).__init__(env, f_se, f_actor, f_critic, f_noise, episode_n, discount_factor,
                                               batch_size, dim_noise, noise_stddev, noise_weight,
                                               ou_params, network_optimizer_ctor,
                                               target_sync_interval, target_sync_rate, replay_size)
Experiment.register(NoisyDPGPendulum, "Noisy DPG explore for pendulum")


class NoisyDPGBipedal(NoisyDPGPendulum):
    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, f_noise=None,
                 episode_n=5000,
                 discount_factor=0.99, batch_size=32, dim_se=16, dim_hidden=64, dim_noise=2,
                 noise_stddev=hrl.utils.CappedLinear(5e5, 0.2, 0.05),
                 noise_weight=1e-1,
                 ou_params=(0.0, 0.2, 0.2),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), target_sync_interval=10,
                 target_sync_rate=0.01, replay_size=10000):
        if env is None:
            env = gym.make("BipedalWalker-v2")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        dim_action = env.action_space.shape[0]
        l2 = 1e-5
        activation = tf.nn.elu
        if f_actor is None:
            def f(inputs):
                input_state = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_state, [dim_hidden], dim_action,
                                                  activation,
                                                  tf.nn.tanh,
                                                  # hrl.tf_dependent.ops.atanh,
                                                  l2=l2)
                return {"action": out}

            f_actor = f

        super(NoisyDPGBipedal, self).__init__(env, f_se, f_actor, f_critic, f_noise, episode_n, discount_factor,
                                              batch_size, dim_se, dim_hidden, dim_noise,
                                              noise_stddev, noise_weight, ou_params,
                                              network_optimizer_ctor, target_sync_interval, target_sync_rate,
                                              replay_size)
Experiment.register(NoisyDPGBipedal, "Noisy DPG explore for bipedal")


if __name__ == '__main__':
    Experiment.main()
