#
# -*- coding: utf-8 -*-

import sys

from hobotrl.experiment import GridSearch, ParallelGridSearch

sys.path.append(".")
import logging

import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.environments import *
import hobotrl.utils as utils
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


def f_dqn_atari(num_action, is_training=False):
    l2 = 1e-8
    def f(inputs):
        input_var = inputs[0]
        print "input size:", input_var
        out = hrl.utils.Network.conv2ds(input_var, shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], out_flatten=True,
                                        activation=tf.nn.relu,
                                        l2=l2, var_scope="convolution")
        out = hrl.utils.Network.layer_fcs(input_var=out, shape=[256], out_count=num_action,
                                          activation_hidden=tf.nn.relu,
                                          activation_out=None,
                                          l2=l2, var_scope="fc")
        return {"q": out}
    return f


def full_wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, max_len=1)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, 4)
    # env = ClippedRewardsWrapper(env)
    return env


class DQNAtari(DQNExperiment):

    def __init__(self, env, f_create_q, episode_n=1000,
                 discount_factor=0.99, ddqn=False, target_sync_interval=100,
                 target_sync_rate=1.0, update_interval=4,
                 replay_size=5000, batch_size=32,
                 greedy_epsilon=utils.CappedLinear(5e5, 0.1, 0.01),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        super(DQNAtari, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                       target_sync_rate, update_interval, replay_size, batch_size, greedy_epsilon,
                                       network_optimizer_ctor)


class DQNPong(DQNAtari):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """

    def __init__(self,
                 greedy_epsilon=utils.CappedLinear(5e5, 0.1, 0.01),
                 learning_rate=1e-3
                 ):
        env = gym.make("PongNoFrameskip-v4")
        env = full_wrap_dqn(env)
        f = f_dqn_atari(env.action_space.n)
        network_optimizer_ctor = lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(learning_rate),
                                                                    grad_clip=10.0)
        super(DQNPong, self).__init__(env, f, greedy_epsilon=greedy_epsilon,
                                      network_optimizer_ctor=network_optimizer_ctor)
Experiment.register(DQNPong, "DQN for Pong")


class GridSearchPong(ParallelGridSearch):

    def __init__(self):
        super(GridSearchPong, self).__init__(DQNPong, {
            "learning_rate": [1e-3, 1e-5],
            "greedy_epsilon": [utils.CappedLinear(2e5, 0.1, 0.01), utils.CappedLinear(5e5, 0.3, 0.01)]
        })
Experiment.register(GridSearchPong, "DQN for Pong")


class PERPong(PERDQNExperiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """
    def __init__(self):
        env = gym.make("PongNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        f = f_dqn_atari(env.action_space.n)
        super(PERPong, self).__init__(env, f)

Experiment.register(PERPong, "PERDQN for Pong")


def f_a3c(num_action):
    l2 = 1e-7

    def create_ac_atari(inputs):
        input_state = inputs[0]
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

    def create_duel_ac_atari(inputs):
        input_state = inputs[0]
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
    return create_ac_atari


class A3CPong(A3CExperiment):
    """
    converges on Pong.
    """

    def __init__(self, env, f_create_net, episode_n=1000, learning_rate=1e-4, discount_factor=0.9, entropy=1e-2,
                 batch_size=8):
        env = gym.make("PongNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        f_create_net = f_a3c(env.action_space.n)
        super(A3CPong, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                      batch_size)
Experiment.register(A3CPong, "Actor Critic for Pong")


class DQNBattleZone(DQNExperiment):
    def __init__(self):
        env = gym.make("BattleZoneNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        f = f_dqn_atari(env.action_space.n)
        super(DQNBattleZone, self).__init__(env, f)

Experiment.register(DQNBattleZone, "DQN for BattleZone")


class DQNBreakout(DQNExperiment):
    def __init__(self):
        env = gym.make("BreakoutNoFrameskip-v4")
        env = ScaledFloatFrame(wrap_dqn(env))
        f = f_dqn_atari(env.action_space.n)
        super(DQNBreakout, self).__init__(env, f)

Experiment.register(DQNBreakout, "DQN for Breakout")


if __name__ == '__main__':
    Experiment.main()
