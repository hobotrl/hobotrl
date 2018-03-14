#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import tensorflow as tf
import numpy as np
import gym

from hobotrl.experiment import ParallelGridSearch, Experiment
import hobotrl.environments as envs
import hobotrl.network as network
import hobotrl.utils as utils
from playground.mpc import MPCAgent


class MPCExperiment(Experiment):

    def __init__(self, env,
                 f_model,
                 sample_n,
                 horizon_n,
                 episode_n=1000,
                 discount_factor=0.99,
                 # sampler arguments
                 update_interval=4,
                 replay_size=1000,
                 batch_size=32,
                 # epsilon greedy arguments
                 greedy_epsilon=0.3,
                 network_optimizer_ctor=lambda: network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0)
                 ):
        self._env, self._f_model, self._episode_n, \
            self._discount_factor, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._greedy_epsilon, \
            self._network_optimizer_ctor = \
            env, f_model, episode_n, \
            discount_factor, \
            update_interval, \
            replay_size, \
            batch_size, \
            greedy_epsilon, \
            network_optimizer_ctor
        self._sample_n, self._horizon_n = sample_n, horizon_n

        super(MPCExperiment, self).__init__()

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        action_shape = list(self._env.action_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = MPCAgent(
            f_model=self._f_model,
            sample_n=self._sample_n,
            horizon_n=self._horizon_n,
            dim_state=state_shape[0],
            dim_action=action_shape[0],
            greedy_epsilon=self._greedy_epsilon,
            # sampler arguments
            update_interval=self._update_interval,
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


class MPCPendulum(MPCExperiment):
    def __init__(self, env=None, f_model=None,
                 sample_n=16, horizon_n=4,
                 episode_n=1000, discount_factor=0.99, update_interval=4,
                 replay_size=100000, batch_size=32,
                 greedy_epsilon=utils.CappedExp(1e5, 2.5, 0.05),
                 network_optimizer_ctor=lambda: network.LocalOptimizer(tf.train.AdamOptimizer(1e-4), grad_clip=10.0)):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1)
        if f_model is None:

            dim_action = env.action_space.shape[0]
            dim_state = env.observation_space.shape[0]
            l2 = 1e-5

            def f(inputs):
                state, action = inputs[0], inputs[1]
                se = tf.concat([state, action], axis=-1)
                se = network.Utils.layer_fcs(se, [256], 256, activation_out=None, l2=l2, var_scope="se")
                goal = network.Utils.layer_fcs(se, [], dim_state, activation_out=None, l2=l2, var_scope="goal")
                reward = network.Utils.layer_fcs(se, [], 1, activation_out=None, l2=l2, var_scope="reward")
                reward = tf.squeeze(reward, axis=1)
                return {"goal": goal, "reward": reward}
            f_model = f

        super(MPCPendulum, self).__init__(env, f_model, sample_n, horizon_n, episode_n, discount_factor,
                                          update_interval, replay_size, batch_size, greedy_epsilon,
                                          network_optimizer_ctor)
Experiment.register(MPCPendulum, "MPC for Pendulum")


class MPCPendulumSearch(ParallelGridSearch):

    def __init__(self, parallel=4):
        parameters = {
            "sample_n": [4, 8, 16],
            "horizon_n": [2, 4, 8],
        }
        super(MPCPendulumSearch, self).__init__(MPCPendulum, parameters, parallel)
Experiment.register(MPCPendulumSearch, "search for MPC for Pendulum")


if __name__ == '__main__':
    Experiment.main()
