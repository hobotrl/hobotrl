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
from hobotrl.tf_dependent.ops import frame_trans
from hobotrl.playback import Playback, BigPlayback
from hobotrl.network import Utils
sys.path.append("../playground/initialD/")
from playground.initialD.ros_environments.clients import DrSimDecisionK8S

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
                 update_interval=8, replay_size=100000, batch_size=8,
                 lower_weight=4.0, upper_weight=4.0, neighbour_size=8,
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
            bucket_size = 8
            traj_count = replay_size / max_traj_length
            bucket_count = traj_count / bucket_size
            active_bucket = 4
            ratio = 1.0 * active_bucket / bucket_count
            transition_epoch = 8
            trajectory_epoch = transition_epoch * max_traj_length
            memory = BigPlayback(
                bucket_cls=Playback,
                bucket_size=bucket_size,
                max_sample_epoch=trajectory_epoch,
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
                                              sampler_creator=f)
Experiment.register(AOTDQNCarRacing, "Asynchronuous OTDQN for CarRacing, tuned with ddqn, duel network, etc.")


class I2A(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 episode_n=10000, learning_rate=1e-4, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4), batch_size=32):
        if env is None:
            # env = gym.make('CarRacing-v0')
            env = DrSimDecisionK8S()
            # env = wrap_car(env, 3, 3)

        if (f_tran and f_rollout and f_ac) is None:
            dim_action = env.action_space.n
            dim_observation = env.observation_space.shape
            chn_se_2d = 32
            dim_se = 256
            nonlinear = tf.nn.elu
            def create_se(inputs):
                l2 = 1e-7
                input_observation = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_observation,

                                               shape=[(8, 8, 4), (16, 4, 2), (chn_se_2d, 3, 2)],
                                               out_flatten=True,
                                               activation=nonlinear,
                                               l2=l2,
                                               var_scope="se_conv")

                se_linear = hrl.utils.Network.layer_fcs(se_conv, [256], dim_se,
                                                        activation_hidden=nonlinear,
                                                        activation_out=None,
                                                        l2=l2,
                                                        var_scope="se_linear")
                return {"se": se_linear}

            def create_ac(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                v = hrl.utils.Network.layer_fcs(input_state, [256, 256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(input_state, [256, 256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                return {"v": v, "pi": pi}

            def create_rollout(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                # rollout that imitates the A3C policy

                rollout_action = hrl.utils.Network.layer_fcs(input_state, [256, 256], dim_action,
                                                             activation_hidden=tf.nn.relu,
                                                             activation_out=tf.nn.softmax,
                                                             l2=l2,
                                                             var_scope="pi")
                return {"rollout_action": rollout_action}

            def create_transition(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                input_action = inputs[1]
                # input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)

                fc_action = hrl.utils.Network.layer_fcs(input_action, [], 256,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_action")

                concat = tf.multiply(input_state, fc_action)
                # concat = tf.concat([input_state, fc_action], axis=-1)

                fc_out = hrl.utils.Network.layer_fcs(concat, [], 256,
                                                     activation_hidden=tf.nn.relu,
                                                     activation_out=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="fc_out")

                # reward
                reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                     activation_hidden=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_state
                next_state = hrl.utils.Network.layer_fcs(fc_out, [256], 256,
                                                         activation_hidden=tf.nn.relu,
                                                         activation_out=None,
                                                         l2=l2,
                                                         var_scope="next_state")

                return {"next_state": next_state, "reward": reward}

            def create_transition_momentum(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                input_action = inputs[1]
                # input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)

                fc_action = hrl.utils.Network.layer_fcs(input_action, [], 256,
                                                        activation_hidden=nonlinear,
                                                        activation_out=nonlinear,
                                                        l2=l2,
                                                        var_scope="fc_action")

                concat = tf.multiply(input_state, fc_action)

                fc_out = hrl.utils.Network.layer_fcs(concat, [], 256,
                                                     activation_hidden=nonlinear,
                                                     activation_out=nonlinear,

                                                     l2=l2,
                                                     var_scope="fc_out")

                # reward
                reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                     activation_hidden=nonlinear,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_goal
                Action_related_goal = hrl.utils.Network.layer_fcs(fc_out, [256], dim_se,
                                                     activation_hidden=nonlinear,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="TC")

                Action_unrelated_goal = hrl.utils.Network.layer_fcs(input_state, [256], dim_se,
                                                     activation_hidden=nonlinear,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="TM")
                Action_unrelated_goal = Utils.scale_gradient(Action_unrelated_goal, 1e-3)
                next_goal = Action_related_goal + Action_unrelated_goal

                return {"next_state": next_goal, "reward": reward, "momentum": Action_unrelated_goal,
                        "action_related": Action_related_goal}

            def create_decoder(inputs):
                l2 = 1e-7
                input_goal = inputs[0]

                input_frame = inputs[1]

                conv_1 = hrl.utils.Network.conv2ds(input_frame,
                                                   shape=[(32, 8, 4)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_1")
                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(64, 4, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")

                conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                                   shape=[(32, 3, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_3")

                conv_3_shape = conv_3.shape.as_list()
                fc_1 = hrl.utils.Network.layer_fcs(input_goal, [], conv_3_shape[1] * conv_3_shape[2] *
                                                   (2 * 256 / conv_3_shape[1] / conv_3_shape[2]),
                                                   activation_hidden=tf.nn.relu,
                                                   activation_out=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="fc_goal")

                twoD_out = tf.reshape(fc_1, [-1, conv_3_shape[1], conv_3_shape[2],
                                             2 * 256 / conv_3_shape[1] / conv_3_shape[2]])

                concat_0 = tf.concat([conv_3, twoD_out], axis=3)

                conv_4 = hrl.utils.Network.conv2ds(concat_0,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_5")

                up_1 = tf.image.resize_images(conv_4, conv_2.shape.as_list()[1:3])

                concat_1 = tf.concat([conv_2, up_1], axis=3)
                concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                     shape=[(32, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_1")

                up_2 = tf.image.resize_images(concat_1, conv_1.shape.as_list()[1:3])

                concat_2 = tf.concat([conv_1, up_2], axis=3)
                concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                     shape=[(64, 4, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_2")

                up_3 = tf.image.resize_images(concat_2, input_frame.shape.as_list()[1:3])

                concat_3 = tf.concat([input_frame, up_3], axis=3)
                concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(64, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_3")

                next_frame = hrl.utils.Network.conv2ds(concat_3,
                                                       shape=[(3, 3, 1)],
                                                       out_flatten=False,
                                                       activation=tf.nn.relu,
                                                       l2=l2,
                                                       var_scope="next_frame")

                return {"next_frame": next_frame}

            def create_decoder_deform(inputs):
                l2 = 1e-7
                input_goal = inputs[0]

                input_frame = inputs[1]

                conv_1 = hrl.utils.Network.conv2ds(input_frame,
                                                   shape=[(8, 4, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_1")
                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")
                se0, goal = tf.split(input_goal, 2, axis=1)
                twoD_out = tf.concat((tf.reshape(se0, [-1, 5, 5, dim_se]),
                                      tf.reshape(goal, [-1, 5, 5, dim_se])), axis=-1)
                conv_5 = hrl.utils.Network.conv2ds(twoD_out,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_5")

                up_1 = tf.image.resize_images(conv_5, [((dim_observation[0] + 3) / 4 + 1) / 2,
                                                       ((dim_observation[1] + 3) / 4 + 1) / 2])

                concat_1 = tf.concat([conv_2, up_1], axis=3)
                concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                     shape=[(16, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_1")

                up_2 = tf.image.resize_images(concat_1, [(dim_observation[0] + 3) / 4, (dim_observation[1] + 3) / 4])

                concat_2 = tf.concat([conv_1, up_2], axis=3)
                concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                     shape=[(8, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_2")

                up_3 = tf.image.resize_images(concat_2, [dim_observation[0], dim_observation[1]])

                concat_3 = tf.concat([input_frame, up_3], axis=3)
                concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(8, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_3")

                next_frame_move = hrl.utils.Network.conv2ds(concat_3,
                                                       shape=[(2, 3, 1)],
                                                       out_flatten=False,
                                                       activation=None,
                                                       l2=l2,
                                                       var_scope="next_frame_move")
                next_frame_move = tf.tanh(next_frame_move) * 16  # within 16 pixels range
                next_frame = frame_trans(input_frame, next_frame_move)
                out = {"next_frame": next_frame}
                return out

            def create_decoder_deform_refine(inputs):
                l2 = 1e-7
                input_goal = inputs[0]

                input_frame = inputs[1]
                pixel_range = 16.0
                gradient_scale = 16.0
                # /2
                conv_1 = hrl.utils.Network.conv2ds(input_frame,
                                                   shape=[(8, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_1")
                # /4
                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_2")
                # /8
                conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_3")

                se0, goal = tf.split(input_goal, 2, axis=1)
                goal = hrl.utils.Network.layer_fcs(input_goal, [256], 6 * 6 * chn_se_2d,
                                                   activation_hidden=nonlinear,
                                                   activation_out=nonlinear,
                                                   l2=l2, var_scope="fc1")
                twoD_out = tf.reshape(goal, [-1, 6, 6, chn_se_2d])
                conv_se = hrl.utils.Network.conv2ds(twoD_out,
                                                   shape=[(chn_se_2d, 3, 1)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_se")

                up_1 = tf.image.resize_images(conv_se, conv_3.shape.as_list()[1:3])

                concat_1 = tf.concat([conv_3, up_1], axis=3)
                move_8 = hrl.utils.Network.conv2ds(concat_1,
                                                     shape=[(2, 3, 1)],
                                                     out_flatten=False,
                                                     activation=None,
                                                     l2=l2,
                                                     var_scope="move_8")
                move_8 = tf.tanh(move_8 / gradient_scale) * (pixel_range / 8)
                up_2 = hrl.network.Utils.deconv(concat_1, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_2")
                up_move_8 = tf.image.resize_images(move_8, conv_2.shape.as_list()[1:3])
                concat_2 = tf.concat([conv_2, up_2, up_move_8 * 2], axis=3)
                move_4 = hrl.utils.Network.conv2ds(concat_2,
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="move_4")
                move_4 = tf.tanh(move_4 / gradient_scale) * (pixel_range / 4)
                up_3 = hrl.network.Utils.deconv(concat_2, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_3")
                up_move_4 = tf.image.resize_images(move_4, conv_1.shape.as_list()[1:3])
                concat_3 = tf.concat([conv_1, up_3, up_move_4 * 2], axis=3)
                move_2 = hrl.utils.Network.conv2ds(concat_3,
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="move_2")
                move_2 = tf.tanh(move_2 / gradient_scale) * (pixel_range / 2)
                up_4 = hrl.network.Utils.deconv(concat_3, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_4")
                frame_shape = input_frame.shape.as_list()[1:3]
                up_move_2 = tf.image.resize_images(move_2, frame_shape)
                concat_4 = tf.concat([input_frame, up_4, up_move_2 * 2], axis=3)
                move_1 = hrl.utils.Network.conv2ds(concat_4,
                                                       shape=[(2, 3, 1)],
                                                       out_flatten=False,
                                                       activation=None,
                                                       l2=l2,
                                                       var_scope="next_frame_move")
                move_1 = tf.tanh(move_1 / gradient_scale) * pixel_range  # within 16 pixels range
                # weighted sum of different scale
                moves = [(move_1, 16.0),
                         (tf.image.resize_images(move_2, frame_shape) * 2, 4.0),
                         (tf.image.resize_images(move_4, frame_shape) * 4, 1.0),
                         (tf.image.resize_images(move_8, frame_shape) * 8, 0.25)]
                sum_weight = sum([m[1] for m in moves])
                move = tf.add_n([m[0] * m[1] for m in moves]) / sum_weight
                next_frame = frame_trans(input_frame, move)
                out = {"next_frame": next_frame}
                return out


            def create_env_upsample_little(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_state = tf.squeeze(tf.stack(input_state), axis=0)

                input_action = inputs[1]
                input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)
                input_action_tiled = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                                                      [((((dim_observation[0]+1)/2+1)/2+1)/2+1)/2,
                                                       ((((dim_observation[1]+1)/2+1)/2+1)/2+1)/2])

                conv_1 = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="conv_1")

                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(64, 4, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")

                conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                                   shape=[(64, 3, 2)],
                                                   out_flatten=True,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_3")

                fc_1 = hrl.utils.Network.layer_fcs(conv_3, [], 64*5*5,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_1")

                # concat_action = tf.concat([conv_4, input_action_tiled], axis=3)
                fc_action = hrl.utils.Network.layer_fcs(tf.to_float(input_action), [], 64*5*5,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_action")

                concat = tf.multiply(fc_1, fc_action)

                fc_out = hrl.utils.Network.layer_fcs(concat, [64*5*5], 64*5*5,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_out")

                # reward
                reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                     activation_hidden=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_state
                twoD_out = tf.reshape(fc_out, [-1, 64, 5, 5])

                conv_5 = hrl.utils.Network.conv2ds(twoD_out,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_5")

                up_1 = tf.image.resize_images(conv_5, [((dim_observation[0]+3)/4+1)/2, ((dim_observation[1]+3)/4+1)/2])

                concat_1 = tf.concat([conv_2, up_1], axis=3)
                concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="concat_1")

                up_2 = tf.image.resize_images(concat_1, [(dim_observation[0]+3)/4, (dim_observation[1]+3)/4])

                concat_2 = tf.concat([conv_1, up_2], axis=3)
                concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                   shape=[(64, 4, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="concat_2")

                up_3 = tf.image.resize_images(concat_2, [dim_observation[0], dim_observation[1]])

                concat_3 = tf.concat([input_state, up_3], axis=3)
                concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(64, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_3")

                next_state = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(3, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="next_state")

                return {"next_state": next_state, "reward": reward}

            def create_encoder(inputs):
                l2 = 1e-7
                input_argu = inputs[0]
                # input_reward = inputs[1]
                #
                # rse = hrl.utils.Network.conv2ds(input_state,
                #                                 shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                #                                 out_flatten=True,
                #                                 activation=tf.nn.relu,
                #                                 l2=l2,
                #                                 var_scope="rse")
                #
                # re_conv = hrl.utils.Network.layer_fcs(rse, [], 200,
                #                                       activation_hidden=tf.nn.relu,
                #                                       activation_out=tf.nn.relu,
                #                                       l2=l2,
                #                                       var_scope="re_conv")
                #
                # # re_conv = tf.concat([re_conv, tf.reshape(input_reward, [-1, 1])], axis=1)
                # re_conv = tf.concat([re_conv, input_reward], axis=1)

                re = hrl.utils.Network.layer_fcs(input_argu, [256], 256,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="re")

                return {"re": re}

            f_se = create_se
            f_ac = create_ac
            # f_env = create_env_upsample_little
            f_rollout = create_rollout
            f_encoder = create_encoder
            f_tran = create_transition_momentum
            f_decoder = create_decoder
            # f_decoder = create_decoder_deform_refine

        super(I2A, self).__init__(env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)
Experiment.register(I2A, "A3C with I2A for CarRacing")


if __name__ == '__main__':
    Experiment.main()
