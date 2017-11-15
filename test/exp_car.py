#
# -*- coding: utf-8 -*-


import sys
sys.path.append(".")
import logging
import numpy as np
import gym
import cv2
import matplotlib.colors as colors
from exp_algorithms import *
from car import *
from hobotrl.tf_dependent.ops import atanh
from hobotrl.environments.environments import *
from hobotrl.playback import Playback, BigPlayback


class A3CCarExp(ACOOExperiment):
    def __init__(self, env, f_create_net=None,
                 episode_n=10000,
                 reward_decay=0.99,
                 on_batch_size=32,
                 off_batch_size=32,
                 off_interval=0,
                 sync_interval=1000,
                 replay_size=128,
                 prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-3),
                 l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(1e-4), ddqn=False, aux_r=False, aux_d=False):

        def create_ac_car(input_state, num_action, **kwargs):
            se = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se")

            q = hrl.utils.Network.layer_fcs(se, [256], num_action,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="q")
            pi = hrl.utils.Network.layer_fcs(se, [256], num_action,
                                             activation_hidden=tf.nn.relu,
                                             # activation_out=tf.nn.softplus,
                                             l2=l2,
                                             var_scope="pi")

            pi = tf.nn.softmax(pi)
            # pi = pi + prob_min
            # pi = pi / tf.reduce_sum(pi, axis=-1, keep_dims=True)
            r = hrl.utils.Network.layer_fcs(se, [256], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="r")

            return {"pi": pi, "q": q, "se": se, "r": r}
        if f_create_net is None:
            f_create_net = create_ac_car
        logging.warning("before super(A3CCarExp, self).__init__")
        super(A3CCarExp, self).__init__(env, f_create_net, episode_n, reward_decay, on_batch_size, off_batch_size,
                                     off_interval, sync_interval, replay_size, prob_min, entropy, l2, optimizer_ctor,
                                     ddqn, aux_r, aux_d)


class A3CCarDiscrete(A3CCarExp):
    def __init__(self):
        env = gym.make("CarRacing-v0")
        env = wrap_car(env, 3, 3)
        super(A3CCarDiscrete, self).__init__(env)

Experiment.register(A3CCarDiscrete, "discrete A3C for CarRacing")


class A3CCarContinuous(A3CExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=1000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-4, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = CarGrassWrapper(env, grass_penalty=0.5)
            env = CarContinuousWrapper(env)
            env = MaxAndSkipEnv(env, skip=2, max_len=1)
            # env = ProcessFrame96H(env)
            env = FrameStack(env, 4)
            env = ScaledRewards(env, 0.1)
            env = ScaledFloatFrame(env)
        if f_create_net is None:
            dim_action = env.action_space.shape[-1]

            def create_ac_car(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se")

                v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                mean = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                   activation_hidden=tf.nn.relu,
                                                   activation_out=None,
                                                   l2=l2,
                                                   var_scope="mean")
                mean = tf.nn.tanh(mean / 4.0)
                stddev = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                     activation_hidden=tf.nn.relu,
                                                     # activation_out=tf.nn.softplus,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="stddev")
                # stddev = 4.0 * tf.nn.sigmoid(stddev / 4.0)
                stddev = 2.0 * (1.0 + atanh(stddev / 4.0))
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = create_ac_car
        super(A3CCarContinuous, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                               batch_size)
Experiment.register(A3CCarContinuous, "continuous A3C for CarRacing")


class A3CCarDiscrete2(A3CExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=10000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 2e-2, 5e-3),
                 batch_size=32):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = wrap_car(env, 3, 3)
        if f_create_net is None:
            dim_action = env.action_space.n

            def create_ac_car(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se")

                v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                return {"v": v, "pi": pi}
            f_create_net = create_ac_car
        super(A3CCarDiscrete2, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                              batch_size)
Experiment.register(A3CCarDiscrete2, "continuous A3C for CarRacing")


class DDPGCar(DPGExperiment):
    def __init__(self, env=None, f_net_ddp=None, f_net_dqn=None, episode_n=10000,
                 optimizer_ddp_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-4),
                 optimizer_dqn_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-3), target_sync_rate=0.001,
                 ddp_update_interval=4, ddp_sync_interval=4, dqn_update_interval=4, dqn_sync_interval=4,
                 max_gradient=10.0, ou_params=(0.0, 0.15, hrl.utils.CappedLinear(2e5, 1.0, 0.05)), gamma=0.99, batch_size=32, replay_capacity=10000):

        l2 = 1e-8

        def f_actor(input_state, action_shape, is_training):
            se = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se")

            action = hrl.utils.Network.layer_fcs(se, [256], action_shape[0],
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.tanh,
                                                 l2=l2,
                                                 var_scope="action")
            logging.warning("action:%s", action)
            return action

        def f_critic(input_state, input_action, is_training):
            se = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se")
            se = tf.concat([se, input_action], axis=1)
            q = hrl.utils.Network.layer_fcs(se, [256], 1,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=None,
                                            l2=l2,
                                            var_scope="q")
            q = tf.squeeze(q, axis=1)
            return q
        f_net_dqn = f_critic if f_net_dqn is None else f_net_dqn
        f_net_ddp = f_actor if f_net_ddp is None else f_net_ddp
        if env is None:
            env = gym.make("CarRacing-v0")
            env = CarGrassWrapper(env, grass_penalty=0.5)
            env = CarContinuousWrapper(env)
            env = MaxAndSkipEnv(env, skip=2, max_len=1)
            env = FrameStack(env, 4)
            env = ScaledRewards(env, 0.1)
            env = ScaledFloatFrame(env)
            env = AugmentEnvWrapper(env,reward_decay=gamma)

        super(DDPGCar, self).__init__(env, f_net_ddp, f_net_dqn, episode_n, optimizer_ddp_ctor, optimizer_dqn_ctor,
                                      target_sync_rate, ddp_update_interval, ddp_sync_interval, dqn_update_interval,
                                      dqn_sync_interval, max_gradient, ou_params, gamma, batch_size, replay_capacity)

Experiment.register(DDPGCar, "DDPG for CarRacing")


class DQNCarRacing(DQNExperiment):

    def __init__(self, env=None, f_create_q=None, episode_n=10000, discount_factor=0.99, ddqn=False, target_sync_interval=100,
                 target_sync_rate=1.0,
                 update_interval=400,
                 replay_size=2000,
                 batch_size=32,
                 greedy_epsilon=hrl.utils.CappedLinear(1e6, 1.0, 0.05),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = wrap_car(env, 3, 3)
        if f_create_q is None:
            l2=1e-8

            def f_critic(inputs):
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se")
                q = hrl.utils.Network.layer_fcs(se, [256], env.action_space.n,
                                                activation_hidden=tf.nn.relu,
                                                activation_out=None,
                                                l2=l2,
                                                var_scope="q")
                return {"q": q}
            f_create_q = f_critic
        super(DQNCarRacing, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                           target_sync_rate, update_interval, replay_size, batch_size, greedy_epsilon,
                                           network_optimizer_ctor)

Experiment.register(DQNCarRacing, "DQN for CarRacing, tuned with ddqn, duel network, etc.")


class ADQNCarRacing(ADQNExperiment):
    def __init__(self, env=None, f_create_q=None, episode_n=10000, discount_factor=0.99, ddqn=True, target_sync_interval=100,
                 target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=32,
                 greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.1, 0.05),
                 learning_rate=1e-4):

        if env is None:
            env = gym.make("CarRacing-v0")
            env = wrap_car(env, 3, 3)
        if f_create_q is None:
            dim_action = env.action_space.n
            activation = tf.nn.elu

            def create_q(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=activation,
                                               l2=l2,
                                               var_scope="se")

                v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=activation,
                                                l2=l2,
                                                activation_out=None,
                                                var_scope="v")
                a = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                 activation_hidden=activation,
                                                 activation_out=None,
                                                 l2=l2,
                                                 var_scope="a")
                a = a - tf.reduce_mean(a, axis=-1, keep_dims=True)
                q = v + a
                return {"q": q}

            f_create_q = create_q

        super(ADQNCarRacing, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, update_interval, replay_size, batch_size, greedy_epsilon,
                                            learning_rate)
Experiment.register(ADQNCarRacing, "Asynchronuous DQN for CarRacing, tuned with ddqn, duel network, etc.")


class AOTDQNCarRacing(AOTDQNExperiment):
    def __init__(self, env=None, f_create_q=None, episode_n=1000, discount_factor=0.99, ddqn=True,
                 target_sync_interval=100, target_sync_rate=1.0,
                 update_interval=8, replay_size=10000, batch_size=8,
                 lower_weight=1.0, upper_weight=1.0, neighbour_size=8,
                 greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.1, 0.05),
                 learning_rate=1e-4):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = wrap_car(env, 3, 3)
        if f_create_q is None:
            dim_action = env.action_space.n
            activation = tf.nn.elu

            def create_q(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=activation,
                                               l2=l2,
                                               var_scope="se")

                v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=activation,
                                                l2=l2,
                                                activation_out=None,
                                                var_scope="v")
                a = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                 activation_hidden=activation,
                                                 activation_out=None,
                                                 l2=l2,
                                                 var_scope="a")
                a = a - tf.reduce_mean(a, axis=-1, keep_dims=True)
                q = v + a
                return {"q": q}

            f_create_q = create_q

        max_traj_length = 500
        def f(args):
            bucket_size = 4
            traj_count = replay_size / max_traj_length
            bucket_count = traj_count / bucket_size
            active_bucket = 2
            ratio = 1.0 * active_bucket / bucket_count
            memory = BigPlayback(
                bucket_cls=Playback,
                bucket_size=bucket_size,
                max_sample_epoch=8,
                capacity=traj_count,
                active_ratio=ratio,
                cache_path=os.sep.join([args.logdir, "cache", str(args.index)])
            )
            sampler = sampling.TruncateTrajectorySampler2(memory, replay_size / max_traj_length, max_traj_length, batch_size, neighbour_size, update_interval)
            return sampler

        def f_simple(args):
            sampler = sampling.TruncateTrajectorySampler2(None, replay_size / max_traj_length, max_traj_length, batch_size, neighbour_size, update_interval)
            return sampler

        super(AOTDQNCarRacing, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                              target_sync_rate, update_interval, replay_size, batch_size, lower_weight,
                                              upper_weight, neighbour_size, greedy_epsilon, learning_rate,
                                              sampler_creator=f_simple)
Experiment.register(AOTDQNCarRacing, "Asynchronuous OTDQN for CarRacing, tuned with ddqn, duel network, etc.")


if __name__ == '__main__':
    Experiment.main()
