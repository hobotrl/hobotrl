
import sys
sys.path.append(".")
import logging

import numpy as np
import tensorflow as tf
import gym

from playground.noisy import NoisySD
from playground.noisy_dpg import NoisyDPG
from hobotrl.experiment import Experiment, GridSearch, ParallelGridSearch
from hobotrl.utils import CappedLinear
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
                 act_ac=False,
                 intrinsic_weight=0.0,
                 explore_net=False,
                 abs_goal=True,
                 manager_ac=True,
                 achievable_weight=1e-3,
                 disentangle_weight=1.0,
                 **kwargs
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
        self._act_ac, self._intrinsic_weight, self._explore_net, \
            self._abs_goal, self._manager_ac, self._achievable_weight, self._disentangle_weight = \
            act_ac, intrinsic_weight, explore_net, \
            abs_goal, manager_ac, achievable_weight, disentangle_weight
        self._kwargs = kwargs

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
            act_ac=self._act_ac,
            intrinsic_weight=self._intrinsic_weight,
            explore_net=self._explore_net,
            abs_goal=self._abs_goal,
            manager_ac=self._manager_ac,
            achievable_weight=self._achievable_weight,
            disentangle_weight=self._disentangle_weight,
            global_step=global_step,
            **self._kwargs
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=args.render_once
            )
            return runner.episode(self._episode_n)


class NoisyPendulum(NoisyExperiment):
    def __init__(self, env=None, se_dimension=3, f_se=None, f_manager=None, f_explorer=None, f_ik=None,
                 f_value=None, f_model=None, f_pi=None,
                 episode_n=2000, discount_factor=0.9,
                 noise_dimension=2, manager_horizon=16, manager_interval=1, batch_size=8, batch_horizon=4,
                 noise_stddev=hrl.utils.CappedLinear(1e5, 1.0, 0.05),
                 # noise_stddev=0.3,
                 noise_explore_param=(0, 0.2, 0.2),
                 # worker_explore_param=(0, 0.2, 0.2),
                 worker_explore_param=(0, 0.2, hrl.utils.CappedLinear(2e5, 0.2, 0.01)),
                 worker_entropy=1e-2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0),
                 replay_size=1000,
                 act_ac=False,
                 intrinsic_weight=0.0,
                 explore_net=False,
                 abs_goal=True,
                 manager_ac=True,
                 achievable_weight=1e-2,
                 disentangle_weight=1e-4,
                 **kwargs
                 ):
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
                out = hrl.network.Utils.scale_gradient(out, 1e-1)
                return {"se": out}

            def fd(inputs):
                return {"se": inputs[0]}
            f_se = fd

        if f_manager is None:
            def f(inputs):
                input_state = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_state, [hidden_dim], se_dimension,
                                                  activation, tf.nn.tanh, l2=l2, var_scope="sd")
                stddev = hrl.network.Utils.layer_fcs(input_state, [hidden_dim], se_dimension,
                                                   activation, None, l2=l2, var_scope="stddev")
                stddev = 2.0 * (1.0 + hrl.tf_dependent.ops.atanh(stddev / 4.0))

                return {"sd": out, "stddev": stddev}
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
                stddev = 2.0 * tf.sigmoid(stddev / 4.0)
                return {"mean": mean, "stddev": stddev}
            f_pi = f

        super(NoisyPendulum, self).__init__(env, f_se, f_manager, f_explorer, f_ik, f_value, f_model, f_pi,
                                            episode_n, discount_factor,
                                            noise_dimension, se_dimension, manager_horizon, manager_interval, batch_size, batch_horizon,
                                            noise_stddev, noise_explore_param, worker_explore_param, worker_entropy,
                                            network_optimizer_ctor, replay_size,
                                            act_ac, intrinsic_weight, explore_net, abs_goal, manager_ac, achievable_weight, disentangle_weight,
                                            **kwargs)
Experiment.register(NoisyPendulum, "Noisy explore for pendulum")


class NoisyPendulumSearch(GridSearch):
    """
        round 1:
            "act_ac": [True, False],
            "intrinsic_weight": [0.0, 1.0],
            "explore_net": [True, False],
            "abs_goal": [True, False],
            "manager_ac": [True, False],
            "achievable_weight": [1e-1, 1e-3]

        conclusion: act_ac = False, explore_net = False

        round 2: how to explore via net?
            "abs_goal": [True],
            "act_ac": [False],
            "explore_net": [True],
            "manager_ac": [False],
            "achievable_weight": [1e-1, 1e-3],
            "disentangle_weight": [1.0, 1e-2, 1e-4],
            "noise_stddev": [CappedLinear(1e5, 1.0, 0.1), CappedLinear(1e5, 0.5, 0.05), CappedLinear(1e5, 0.1, 0.01)]

        round 3: manager_ac debugged
            "abs_goal": [True],
            "act_ac": [False],
            "explore_net": [False],
            "manager_ac": [True],
            "achievable_weight": [1e-1, 1e-3],
            "worker_explore_param":  [(0, 0.2, CappedLinear(2e5, 0.2, 0.01)), (0, 0.2, CappedLinear(2e5, 0.5, 0.1))],
            "manager_entropy": [1e-1, 1e-2, CappedLinear(2e5, 1e-1, 1e-2), CappedLinear(2e5, 1e-2, 1e-3)],

        round 4: manager_ac longer
            "episode_n": [5000],
            "abs_goal": [True],
            "act_ac": [False],
            "explore_net": [False],
            "manager_ac": [True],
            "achievable_weight": [1e-3],
            "worker_explore_param":  [(0, 0.2, CappedLinear(2e5, 0.2, 0.01)), (0, 0.2, CappedLinear(2e5, 0.5, 0.1))],
            "manager_entropy": [1e-2, CappedLinear(2e5, 1e-2, 1e-3), CappedLinear(5e5, 1e-2, 1e-4)],
        round 5: manager_ac, with momentum
            "explicit_momentum": [True, False],
            "act_ac": [False],
            "explore_net": [False],
            "manager_ac": [True],
            "manager_entropy": [1e-2],
            "achievable_weight": [1e-3, 1e-4],
            # "worker_explore_param":  [(0, 0.2, CappedLinear(2e5, 0.2, 0.01)), (0, 0.2, CappedLinear(2e5, 0.5, 0.1))],
            "worker_explore_param": [(0, 0.2, CappedLinear(2e5, 0.2, 0.01))],
            "imagine_history": [True, False],
        round 6: imaginary training
            "explicit_momentum": [True],
            "manager_entropy": [1e-2],
            "worker_explore_param":  [(0, 0.2, CappedLinear(2e5, 0.2, 0.01)), (0, 0.2, CappedLinear(3e5, 0.5, 0.05))],
            # "worker_explore_param": [(0, 0.2, CappedLinear(2e5, 0.2, 0.01))],
            "imagine_history": [True],
            "momentum_weight": [1e-1, 1e-2],
            "imagine_weight": [1.0, 1e-1],
            "_r": [0, 1],
        round 7: stability
    """
    def __init__(self):
        super(NoisyPendulumSearch, self).__init__(NoisyPendulum, {
            "achievable_weight": [1e-1, 1e-4],
            "explicit_momentum": [True],
            "manager_entropy": [1e-2],
            "worker_explore_param":  [(0, 0.2, CappedLinear(2e5, 0.2, 0.01))],
            "imagine_history": [True],
            "momentum_weight": [1e-1],
            "imagine_weight": [1.0],
            "_r": [0, 1],
        })
Experiment.register(NoisyPendulumSearch, "Noisy explore for pendulum")


class ParallelNoisyPendulumSearch(ParallelGridSearch):
    def __init__(self):
        super(ParallelNoisyPendulumSearch, self).__init__(NoisyPendulum, {
            "episode_n": [10],
            "achievable_weight": [1e-1, 1e-4],
            "explicit_momentum": [True, False],
            "manager_entropy": [1e-2],
            "worker_explore_param": [(0, 0.2, CappedLinear(2e5, 0.2, 0.01))],
            "imagine_history": [True],
            "momentum_weight": [1e-1],
            "imagine_weight": [1.0, 0.1],
            "_r": [0, 1],
        })
Experiment.register(ParallelNoisyPendulumSearch, "Noisy explore for pendulum")


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
                 noise_stddev_weight=1e-4,
                 noise_mean_weight=1e-2,
                 ou_params=(0.0, 0.2, 0.2),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 replay_size=1000,
                 **kwargs
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
            self._noise_stddev_weight, \
            self._noise_mean_weight, \
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
            noise_stddev_weight, \
            noise_mean_weight, \
            ou_params, \
            network_optimizer_ctor, \
            target_sync_interval, \
            target_sync_rate, \
            replay_size
        self._kwargs = kwargs

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        dim_action = self._env.action_space.shape[0]
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
        logging.warning("ctor:%s", self._network_optimizer_ctor)
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
            noise_stddev_weight=self._noise_stddev_weight,
            noise_mean_weight=self._noise_mean_weight,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            global_step=global_step,
            **self._kwargs
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=args.render_once
            )
            return runner.episode(self._episode_n)


class FDPG(object):

    def __init__(self, dim_action, dim_se=None, activation=tf.nn.elu, l2=1e-5):
        super(FDPG, self).__init__()
        self._dim_action, self._dim_se, self._activation, self._l2 = \
            dim_action, dim_se, activation, l2
        self._dim_hidden = 128

    def se(self):
        def f(inputs):
            input_state = inputs[0]
            out = hrl.network.Utils.layer_fcs(input_state, [], self._dim_se,
                                              None, None, l2=self._l2)
            return {"se": out}

        def i(inputs):
            return {"se": inputs[0]}

        return f if self._dim_se is not None else i

    def actor(self):
        def f(inputs):
            input_state = inputs[0]
            out = hrl.network.Utils.layer_fcs(input_state, [self._dim_hidden], self._dim_action,
                                              self._activation,
                                              # hrl.tf_dependent.ops.atanh,
                                              tf.nn.tanh,
                                              l2=self._l2)
            out = hrl.network.Utils.scale_gradient(out, 1e-3)
            return {"action": out}
        return f

    def noise(self):
        def f(inputs):
            input_se, input_noise = inputs[0], inputs[1]
            input_var = tf.concat([input_se, input_noise], axis=-1)
            out = hrl.network.Utils.layer_fcs(input_var, [self._dim_hidden, self._dim_hidden], self._dim_action,
                                              self._activation,
                                              tf.nn.tanh,
                                              l2=self._l2)
            return {"noise": out}
        return f

    def critic(self):
        def f(inputs):
            input_se, input_action = inputs[0], inputs[1]
            input_var = tf.concat([input_se, input_action], axis=-1)
            out = hrl.network.Utils.layer_fcs(input_var, [self._dim_hidden, self._dim_hidden], 1,
                                              self._activation, None, l2=self._l2)
            out = tf.squeeze(out, axis=-1)
            return {"q": out}
        return f


class NoisyDPGPendulum(NoisyDPGExperiment):
    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, f_noise=None,
                 episode_n=1000, discount_factor=0.9, batch_size=32,
                 dim_se=16, dim_hidden=32, dim_noise=2,
                 # noise_stddev=hrl.utils.CappedLinear(1e5, 0.1, 0.02),
                 noise_stddev=1.0,
                 noise_weight=1e-2,
                 noise_stddev_weight=1e-4,
                 noise_mean_weight=1e-2,
                 ou_params=(0.0, 0.2, 0.2),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), target_sync_interval=10,
                 target_sync_rate=0.01, replay_size=1000,
                 **kwargs):

        if env is None:
            env = gym.make("Pendulum-v0")
            # env = hrl.envs.MaxAndSkipEnv(env, 1, skip=2)
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)

        dim_action = env.action_space.shape[0]
        l2 = 1e-5
        activation = tf.nn.elu
        f = FDPG(dim_action, activation=activation, l2=l2)
        f_se = f.se() if f_se is None else f_se
        f_actor = f.actor() if f_actor is None else f_actor
        f_noise = f.noise() if f_noise is None else f_noise
        f_critic = f.critic() if f_critic is None else f_critic
        super(NoisyDPGPendulum, self).__init__(env, f_se, f_actor, f_critic, f_noise, episode_n, discount_factor,
                                               batch_size, dim_noise, noise_stddev, noise_weight,
                                               noise_stddev_weight, noise_mean_weight,
                                               ou_params, network_optimizer_ctor,
                                               target_sync_interval, target_sync_rate, replay_size, **kwargs)
Experiment.register(NoisyDPGPendulum, "Noisy DPG explore for pendulum")


class NoisyDPGBipedal(NoisyDPGPendulum):
    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, f_noise=None,
                 episode_n=4000,
                 discount_factor=0.99, batch_size=32, dim_se=16, dim_hidden=64, dim_noise=2,
                 noise_stddev=hrl.utils.CappedLinear(5e5, 0.2, 0.05),
                 noise_weight=1e-1,
                 noise_stddev_weight=1e-4,
                 noise_mean_weight=1e-2,
                 ou_params=(0.0, 0.2, 0.2),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), target_sync_interval=10,
                 target_sync_rate=0.01, replay_size=10000, **kwargs):
        if env is None:
            env = gym.make("BipedalWalker-v2")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        # dim_action = env.action_space.shape[0]
        # l2 = 1e-5
        # activation = tf.nn.elu
        # if f_actor is None:
        #     def f(inputs):
        #         input_state = inputs[0]
        #         out = hrl.network.Utils.layer_fcs(input_state, [dim_hidden], dim_action,
        #                                           activation,
        #                                           tf.nn.tanh,
        #                                           # hrl.tf_dependent.ops.atanh,
        #                                           l2=l2)
        #         return {"action": out}
        #
        #     f_actor = f

        super(NoisyDPGBipedal, self).__init__(env, f_se, f_actor, f_critic, f_noise, episode_n, discount_factor,
                                              batch_size, dim_se, dim_hidden, dim_noise,
                                              noise_stddev, noise_weight,
                                              noise_stddev_weight, noise_mean_weight,
                                              ou_params,
                                              network_optimizer_ctor, target_sync_interval, target_sync_rate,
                                              replay_size, **kwargs)
Experiment.register(NoisyDPGBipedal, "Noisy DPG explore for bipedal")


class NoisyDPGBipedalSearch(ParallelGridSearch):
    """
    round 1
    [
            {
                "noise_stddev": [hrl.utils.CappedLinear(5e5, 0.2, 0.05), hrl.utils.CappedLinear(1e6, 0.5, 0.02)],
                "noise_weight": [1e-1, 1e-3],
            },
            {
                "noise_stddev": [hrl.utils.CappedLinear(5e5, 1.0, 0.5)],
                "noise_weight": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            }
        ]
    round 2
            {
                "noise_stddev": [hrl.utils.CappedLinear(5e5, 0.2, 0.05), hrl.utils.CappedLinear(2e6, 0.2, 0.05)],
                "noise_weight": [1e-1, 1e-2, 1e-3, 1e-4],
            }
    round 3
            {
                "noise_stddev": [hrl.utils.CappedLinear(5e5, 0.2, 0.02),
                                 hrl.utils.CappedLinear(5e5, 0.1, 0.02),
                                 hrl.utils.CappedLinear(5e5, 0.2, 0.05)],
                "noise_weight": [1e-1],
            }
    round 4
            {
                "noise_stddev_weight": [1e-2, 1e-3, 1e-4, 1e-5],
                "noise_mean_weight": [1e-1, 1e-3, 1e-5],
            }

    """
    def __init__(self):
        super(NoisyDPGBipedalSearch, self).__init__(NoisyDPGBipedal,
        [
            {
                "noise_stddev": [hrl.utils.CappedLinear(5e5, 0.2, 0.02),
                                 hrl.utils.CappedLinear(5e5, 0.1, 0.02),
                                 hrl.utils.CappedLinear(5e5, 0.2, 0.05)],
                "noise_weight": [1e-1, 1e-3],
                "episode_n": [5000],
                "disentangle_with_dpg": [False]
            }
        ], parallel=4)

Experiment.register(NoisyDPGBipedalSearch, "Noisy DPG explore for bipedal")


if __name__ == '__main__':
    Experiment.main()
