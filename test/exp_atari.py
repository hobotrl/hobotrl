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

from hobotrl.environments import *
from exp_algorithms import *


def state_trans_atari(state):
    gray = np.asarray(np.dot(state, [0.299, 0.587, 0.114]))
    gray = cv2.resize(gray, (84, 84))
    return np.asarray(gray.reshape(gray.shape + (1,)), dtype=np.uint8)


def state_trans_atari2(state):
    img = state
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    return np.asarray(resized_screen.reshape(resized_screen.shape + (1,)), dtype=np.uint8)


def state_trans_atari3(state):
    gray = np.asarray(np.dot(state, [0.299, 0.587, 0.114]))
    resized_screen = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized_screen.reshape(resized_screen.shape + (1,)) / 255.0


def f_dqn_atari(inputs, num_action, is_training):
    input_var = inputs
    print "input size:", input_var
    out = hrl.utils.Network.conv2ds(input_var, shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], out_flatten=True,
                                    activation=tf.nn.relu,
                                    l2=1e-8, var_scope="convolution")
    out = hrl.utils.Network.layer_fcs(input_var=out, shape=[256], out_count=num_action,
                                      activation_hidden=tf.nn.relu,
                                      activation_out=None,
                                      l2=1e-8, var_scope="fc")
    return out


class DQNAtari(DQNExperiment):
    def __init__(self, env, f_create_net=None,
                 episode_n=10000,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-4),
                 target_sync_rate=1.0,
                 update_interval=4,
                 target_sync_interval=1000,
                 max_gradient=10.0,
                 epsilon=hrl.utils.CappedLinear(step=4e5, start=1.0, end=0.01),
                 gamma=0.99,
                 greedy_policy=True,
                 ddqn=False,
                 batch_size=32,
                 replay_capacity=10000):

        if f_create_net is None:
            f_create_net = f_dqn_atari
        super(DQNAtari, self).__init__(env, f_create_net, episode_n, optimizer_ctor, target_sync_rate, update_interval,
                               target_sync_interval, max_gradient, epsilon, gamma, greedy_policy, ddqn, batch_size,
                               replay_capacity)


class PERAtari(PERDQNExperiment):
    def __init__(self, env, f_create_net=None,
                 episode_n=1000,
                 priority_bias=0.5,
                 importance_weight=hrl.utils.CappedLinear(1e6, 0.5, 1.0),
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-4),
                 target_sync_rate=1.0,
                 update_interval=4,
                 target_sync_interval=1000,
                 max_gradient=10.0,
                 epsilon=hrl.utils.CappedLinear(step=4e5, start=1.0, end=0.01),
                 gamma=0.99,
                 greedy_policy=True,
                 ddqn=False,
                 batch_size=32,
                 replay_capacity=10000):
        if f_create_net is None:
            f_create_net = f_dqn_atari
        super(PERAtari, self).__init__(env, f_create_net, episode_n, priority_bias, importance_weight, optimizer_ctor,
                                       target_sync_rate, update_interval, target_sync_interval, max_gradient, epsilon,
                                       gamma, greedy_policy, ddqn, batch_size, replay_capacity)


class DQNPong(DQNAtari):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """

    def __init__(self):
        env = gym.make("PongNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        super(DQNPong, self).__init__(env)

Experiment.register(DQNPong, "DQN for Pong")


class PERPong(PERAtari):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """
    def __init__(self):
        env = gym.make("PongNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        super(PERPong, self).__init__(env)

Experiment.register(PERPong, "PERDQN for Pong")


class ACOOAtari(ACOOExperiment):

    def __init__(self, env, f_create_net=None, episode_n=10000, reward_decay=0.99, on_batch_size=32, off_batch_size=32,
                 off_interval=8, sync_interval=1000, replay_size=10000, prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(4e5, 1e-2, 1e-3), l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(1e-4), ddqn=False, aux_r=False, aux_d=False):

        def create_ac_atari(input_state, num_action, **kwargs):
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

        def create_duel_ac_atari(input_state, num_action, **kwargs):
            with tf.variable_scope("se"):
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="convolution")
                se = hrl.utils.Network.layer_fcs(se, [], 256, activation_hidden=None,
                                                 activation_out=tf.nn.relu, l2=l2, var_scope="fc")

                a = hrl.utils.Network.layer_fcs(se, [], num_action, activation_hidden=None,
                                                activation_out=None, l2=l2, var_scope="a")
                a = a - tf.reduce_mean(a, axis=-1, keep_dims=True)

            with tf.variable_scope("q"):
                v = hrl.utils.Network.layer_fcs(se, [], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                q = v + a

            pi = hrl.utils.Network.layer_fcs(a, [], num_action,
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
            f_create_net = create_ac_atari

        super(ACOOAtari, self).__init__(env, f_create_net, episode_n, reward_decay, on_batch_size, off_batch_size,
                                        off_interval, sync_interval, replay_size, prob_min, entropy, l2, optimizer_ctor,
                                        ddqn, aux_r, aux_d)


class ACOOPong(ACOOAtari):
    """
    converges on Pong.
    """

    def __init__(self):
        env = gym.make("PongNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        ACOOAtari.__init__(self, env)

Experiment.register(ACOOPong, "Actor Critic for Pong")


class DQNBattleZone(DQNAtari):
    def __init__(self):
        env = gym.make("BattleZoneNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        super(DQNBattleZone, self).__init__(env)

Experiment.register(DQNBattleZone, "DQN for BattleZone")


class DQNBreakout(DQNAtari):
    def __init__(self):
        env = gym.make("BreakoutNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        super(DQNBreakout, self).__init__(env)

Experiment.register(DQNBreakout, "DQN for Breakout")


if __name__ == '__main__':
    Experiment.main()
