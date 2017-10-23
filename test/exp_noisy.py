
import sys
sys.path.append(".")

import numpy as np
import tensorflow as tf
import gym

from playground.noisy import NoisySD
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
                 episode_n=1000,
                 discount_factor=0.9,
                 noise_dimension=2,
                 manager_horizon=32,
                 batch_size=32,
                 batch_horizon=4,
                 noise_stddev=1.0,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 ):
        super(NoisyExperiment, self).__init__()
        self._env, self._f_se, self._f_manager, self._f_explorer, self._f_ik, self._f_value, \
            self._episode_n, self._discount_factor, \
            self._noise_dimension, self._manager_horizon, self._batch_size, self._batch_horizon, self._noise_stddev, \
            self._network_optimizer_ctor = \
            env, f_se, f_manager, f_explorer, f_ik, f_value, \
            episode_n, discount_factor, \
            noise_dimension, manager_horizon, batch_size, batch_horizon, noise_stddev, \
            network_optimizer_ctor

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)

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
            state_shape=state_shape,
            noise_dimension=self._noise_dimension,
            discount_factor=self._discount_factor,
            network_optimizer=self._network_optimizer_ctor(),
            noise_stddev=self._noise_stddev,
            manager_horizon=self._manager_horizon,
            batch_size=self._batch_size,
            batch_horizon=self._batch_horizon,
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
    def __init__(self, env=None, se_dimension=4, f_se=None, f_manager=None, f_explorer=None, f_ik=None, f_value=None,
                 episode_n=2000, discount_factor=0.9,
                 noise_dimension=2, manager_horizon=32, batch_size=32, batch_horizon=4,
                 # noise_stddev=hrl.utils.CappedLinear(1e5, 0.5, 0.1),
                 noise_stddev=0.3,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)

        action_dimension = env.action_space.shape[0]
        l2 = 1e-5
        activation = tf.nn.elu
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
                out = hrl.network.Utils.layer_fcs(input_state, [16], se_dimension,
                                                  activation, tf.nn.tanh, l2=l2)
                return {"sd": out}
            f_manager = f

        if f_explorer is None:
            def f(inputs):
                input_se, input_noise = inputs[0], inputs[1]
                input_var = tf.concat([input_se, input_noise], axis=-1)
                out = hrl.network.Utils.layer_fcs(input_var, [16], se_dimension,
                                                  activation, None, l2=l2)
                return {"noise": out}
            f_explorer = f

        if f_ik is None:
            def f(inputs):
                input_se, input_goal = inputs[0], inputs[1]
                input_var = tf.concat([input_se, input_goal], axis=-1)
                out = hrl.network.Utils.layer_fcs(input_var, [16, 16], action_dimension,
                                                  activation, tf.nn.tanh, l2=l2)
                return {"action": out}
            f_ik = f

        if f_value is None:
            def f(inputs):
                input_se = inputs[0]
                out = hrl.network.Utils.layer_fcs(input_se, [16, 16], 1,
                                                  activation, tf.nn.tanh, l2=l2)
                out = tf.squeeze(out, axis=-1)
                return {"value": out}
            f_value = f

        super(NoisyPendulum, self).__init__(env, f_se, f_manager, f_explorer, f_ik, f_value, episode_n, discount_factor,
                                            noise_dimension, manager_horizon, batch_size, batch_horizon, noise_stddev,
                                            network_optimizer_ctor)
Experiment.register(NoisyPendulum, "Noisy explore for pendulum")


if __name__ == '__main__':
    Experiment.main()