#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import hobotrl as hrl
from hobotrl.experiment import Experiment
from hobotrl.environments import *

from playground.dpg_ot import OTDPG


class OTDPGExperiment(Experiment):

    def __init__(self, env,
                 f_se, f_actor, f_critic,
                 lower_weight, upper_weight, neighbour_size,
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
                 batch_size=32,
                 replay_capacity=1000):
        self._env, self._f_se, self._f_actor, self._f_critic, self._episode_n,\
            self._discount_factor, self._network_optimizer_ctor, \
            self._ou_params, self._target_sync_interval, self._target_sync_rate, \
            self._batch_size, self._replay_capacity = \
            env, f_se, f_actor, f_critic, episode_n, \
            discount_factor, network_optimizer_ctor, \
            ou_params, target_sync_interval, target_sync_rate, \
            batch_size, replay_capacity
        self._lower_weight, self._upper_weight, self._neighbour_size = lower_weight, upper_weight, neighbour_size
        super(OTDPGExperiment, self).__init__()

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)
        dim_action = self._env.action_space.shape[-1]
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = OTDPG(
            f_se=self._f_se,
            f_actor=self._f_actor,
            f_critic=self._f_critic,
            lower_weight=self._lower_weight,
            upper_weight=self._upper_weight,
            neighbour_size=self._neighbour_size,
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
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint, render_interval=40,
                render_once=True,
                logdir=args.logdir
            )
            runner.episode(self._episode_n)


class OTDPGPendulum(OTDPGExperiment):
    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None,
                 lower_weight=4, upper_weight=4, neighbour_size=8, episode_n=1000,
                 discount_factor=0.9,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0),
                 ou_params=(0, 0.2, 0.1),
                 target_sync_interval=10, target_sync_rate=0.01, batch_size=8, replay_capacity=1000):
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
        super(OTDPGPendulum, self).__init__(env, f_se, f_actor, f_critic, lower_weight, upper_weight, neighbour_size,
                                            episode_n, discount_factor, network_optimizer_ctor, ou_params,
                                            target_sync_interval, target_sync_rate, batch_size, replay_capacity)
Experiment.register(OTDPGPendulum, "OTDPG for Pendulum")


if __name__ == '__main__':
    Experiment.main()