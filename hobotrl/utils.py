# -*- coding: utf-8 -*-

"""Main building blocks for RL agent.
Guys who do the heavy-lifting lies here. Supposed to be friendly to
both mixin-style and non-... usages.

1. TabularQFunc
2. EpsilonGreedyPolicy
3. ExperienceReplay
"""

import numpy as np
from numpy import max
from numpy.random import rand, randint
import tensorflow.contrib.layers as layers
import tensorflow as tf
from core import BaseAgent


class TabularQFunc(object):
    """Table-based Action-Value Function.
    This class implements the classical table-based action value function
    (i.e. Q function). The Q values are stored in a dictionary for each
    (state, action) pair and can be updated with the temporal-difference (TD)
    learning algorithm.
    """
    def __init__(self, actions, gamma, greedy_policy=True, alpha=1.0,
                 default_q_val=0.0, **kwargs):
        """Initialization

        Parameters
        ----------
        actions : the action space. (Tuple)
        gamma   : discount factor for value functions. (Float)
        greedy_policy : whether or not to evaluate the greedy policy. (Bool)
        alpha   : (optional) 1-alpha is the exponential decay factor for old
                  Q values. (Float)
        default_q_val : (optional) default value for uninitialized action-value
                        entries. (Float)
        """
        self.__ACTIONS = actions
        self.__GAMMA = gamma  # discount factor
        self.__GREEDY_POLICY = greedy_policy
        self.__ALPHA = alpha  # Moving average exponent for T-D updates
        self.__DEFAULT_QVAL = default_q_val

        self.__q = {}

    def get_value(self, state, action=None, **kwargs):
        """Retrieve action-value entries
        Return action-value entry for specified (state, action) pair or for all
        actions of a particular state if the "action" is None.

        Parameters
        ----------
        state  :
        action :
        """
        if action is None:  # return Q values for all actions
            return [
                self.__q[(state, a)] if (state, a) in self.__q else \
                self.__DEFAULT_QVAL
                for a in self.__ACTIONS
            ]
        else:
            exp = (state, action)
            return self.__q[exp] if exp in self.__q else self.__DEFAULT_QVAL

    def improve_value_(self, state, action, reward,
                       next_state, next_action=None,
                       episode_done=False, importance=1.0,
                       **kwargs):
        """Evaluate policy with one-step temporal difference.
        This method evaluate a policy by means of the temporal difference
        algorithm and forms a tabular action-value function.

        Depending on the class attr. "GREEDY_POLICY", this method either
        evaluate the greedy policy (True) or evaluate other policies (False).
        In the latter case, the "importance" arg. can also be provided for
        off-policy evaluation. It will be used to correct the bias on action
        selection.

        Note the "importance" arg. will be ignored in the greedy policy case
        as well as in the on-policy case with default next action.

        Parameters
        ----------
        state  :
        action :
        reward :
        next_state   :
        next_action  :
        episode_done :
        importance   : importance sampling ratio for off-policy evaluation.
                       Use default (1.0) for greedy of on-policy evaluation.
        """
        # Getting the Q value for next step:
        # If evaluate the greedy policy use the maximum Q value across all
        # actions.
        if self.__GREEDY_POLICY:
            # greedy policy suggests an unit importance
            importance = 1.0
            next_q = max(self.get_value(next_state))
        # If evaluate other policies, either use the "next_action" passed in
        # or sample next action with "act_()" if "next_action" is None.
        else:
            next_q = self.get_value(next_state, next_action)

        # Target Q value from Bellman iteration
        target_q = reward + self.__GAMMA * importance * next_q * (1 - episode_done)

        # Standard Temporal Difference update with exponention moving
        # averaging, i.e update is the average of old value and new target.
        exp = (state, action)
        if exp not in self.__q:
            self.__q[exp] = self.__DEFAULT_QVAL
        td = target_q - self.__q[exp]
        self.__q[exp] += self.__ALPHA * td

        return {'td': td}


class EpsilonGreedyPolicy(object):
    """Epsilon greedy policy
    This policy superimpose a random policy onto the greedy policy with a small
    probability epsilon.

    Assume super-class already initialized a discrete and index-able action
    space. And assume a action-value func. mixin already implements the
    "get_value()" method for value retrieval.
    """
    def __init__(self, actions, f_get_value, epsilon, tol=1e-10, **kwargs):
        """Initialization

        Parameters
        ----------
        actions :
        f_get_value :
        epsilon : probability of choosing random action.
        tol     : a small tolerance for equality tests.
        """
        self.__ACTIONS = actions
        self.__get_value = f_get_value
        self.__EPSILON = epsilon
        self.__TOL = tol

    def act(self, state, **kwargs):
        """Epsilon greedy action selection.
        Choose greedy action with 1-epsilon probability and random action with
        epsilon probability. Ties are broken randomly for greedy actions.
        """
        if state is None or rand() < self.__EPSILON:
            idx_action = randint(0, len(self.__ACTIONS))
        else:
            # Follow greedy policy with 1-epsilon prob.
            # break tie randomly
            q_vals = np.asarray(
                self.__get_value(state=state, **kwargs)
            ).flatten()
            if 'print_qval' in kwargs and kwargs['print_qval']:
                print q_vals
            max_q_val = max(q_vals)
            idx_best_actions = [
                i for i in range(len(q_vals))
                if (q_vals[i] - max_q_val)**2 < self.__TOL
            ]
            idx_action = idx_best_actions[randint(0, len(idx_best_actions))]

        return self.__ACTIONS[idx_action]


class Network(object):

    @staticmethod
    def layer_fcs(input_var, shape, out_count, activation_hidden=tf.nn.relu, activation_out=None, l2=0.0001,
                  var_scope=""):
        variables = []
        ops = []
        with tf.variable_scope(var_scope):
            for i in range(len(shape)):
                hidden_count = shape[i]
                out = layers.fully_connected(inputs=input_var, num_outputs=hidden_count,
                                             activation_fn=activation_hidden,
                                             weights_initializer=layers.xavier_initializer(),
                                             biases_initializer=layers.xavier_initializer(),
                                             weights_regularizer=layers.l2_regularizer(l2),
                                             biases_regularizer=layers.l2_regularizer(l2),
                                             scope="hidden_%d" % i)
                input_var = out

            # output
            with tf.variable_scope("out"):
                out = layers.fully_connected(inputs=input_var, num_outputs=out_count,
                                             activation_fn=activation_out,
                                             weights_initializer=layers.xavier_initializer(),
                                             biases_initializer=layers.xavier_initializer(),
                                             weights_regularizer=layers.l2_regularizer(l2),
                                             biases_regularizer=layers.l2_regularizer(l2),
                                             scope="out")
        return out

    @staticmethod
    def conv2d(input_var, h, w, out_channel, strides=[1, 1], padding="VALID",
               activation=tf.nn.relu, l2=1e-4, var_scope=""):
        with tf.variable_scope(var_scope):
            out = tf.layers.conv2d(inputs=input_var, filters=out_channel, kernel_size=[w, h],
                                   strides=strides, padding=padding, activation=activation,
                                   use_bias=True, kernel_initializer=layers.xavier_initializer(),
                                   bias_initializer=layers.xavier_initializer(),
                                   kernel_regularizer=layers.l2_regularizer(l2),
                                   bias_regularizer=layers.l2_regularizer(l2))
        return out

    @staticmethod
    def clipped_square(value, clip=1.0):
        abs_value = tf.abs(value)
        quadratic = tf.minimum(abs_value, clip)
        linear = abs_value - quadratic
        return 0.5 * tf.square(quadratic) + clip * linear


class Sequence(object):
    """
    Sequence value generator, can be used for variable hyperparameter
    """
    def __init__(self, func):
        self.n = 0
        self.func = func

    def __call__(self, *args, **kwargs):
        func = self.func
        ret = func(self.n)
        self.n += 1
        return ret


class LinearSequence(Sequence):
    def __init__(self, step, start, end):
        super(LinearSequence, self).__init__(lambda n: start + (end - start) * n / step)
