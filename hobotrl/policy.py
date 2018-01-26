# -*- coding: utf-8 -*-
import logging

import numpy as np
from network import NetworkFunction
from hobotrl.tf_dependent.distribution import NNDistribution, DiscreteDistribution, NormalDistribution
from core import Policy


class GreedyPolicy(Policy):
    def __init__(self, q_function):
        """
        :param q_function:
        :type q_function: NetworkFunction
        """
        super(GreedyPolicy, self).__init__()
        self.q_function = q_function
        self._num_actions = q_function.output().op.shape.as_list()[-1]

    def act(self, state, **kwargs):
        q_values = self.q_function(np.asarray(state)[np.newaxis, :])[0]
        action = np.argmax(q_values)
        return action


class EpsilonGreedyPolicy(Policy):
    def __init__(self, q_function, epsilon, num_actions):
        """

        :param q_function:
        :type q_function: NetworkFunction
        :param epsilon:
        :param num_actions:
        """
        super(EpsilonGreedyPolicy, self).__init__()
        self.q_function, self._epsilon, self._num_actions = q_function, epsilon, num_actions

    def act(self, state, exploration=True, **kwargs):
        if exploration and np.random.rand() < self._epsilon:
            action = np.random.randint(self._num_actions)
        else:
            q_values = self.q_function(np.asarray(state)[np.newaxis, :])[0]
            action = np.argmax(q_values)
        return action


class OUNoise(object):
    def __init__(self, shape, mu, theta, sigma):
        """
        :param shape:
        :param mu: mean of noise
        :param theta: 1 - momentum of noise
        :param sigma: scale of noise
        """
        self._shape, self._mu, self._theta, self._sigma = shape, mu, theta, sigma
        self._x = np.ones(self._shape) * self._mu

    def tick(self):
        self._x += self._theta * (self._mu - self._x) +\
                   self._sigma * np.random.randn(*self._shape)
        return self._x


class OUExplorationPolicy(Policy):
    def __init__(self, action_function, mu, theta, sigma):
        """

        :param action_function:
        :type action_function: NetworkFunction
        :param mu:
        :param theta:
        :param sigma:
        """
        self._action_function = action_function
        self._action_shape = [action_function.output().op.shape.as_list()[-1]]
        self._ou_noise = OUNoise(self._action_shape, mu, theta, sigma)

    @staticmethod
    def action_add(action, noise):
        """
        regularize action + noise into (-1, 1)
        :param action:
        :param noise:
        :return:
        """
        return action + np.abs(np.sign(noise) - action) * np.tanh(noise)

    def act(self, state, **kwargs):
        action = self._action_function(np.asarray(state)[np.newaxis, :])[0]
        noise = self._ou_noise.tick()
        action0 = self.action_add(action, noise)
        # logging.warning("action: %s + %s -> %s", action, noise, action0)
        return action0


class StochasticPolicy(Policy):
    """
    returns action according to probability distribution.
    """
    def __init__(self, distribution):
        """
        :param distribution:
        :type distribution NNDistribution
        """
        super(StochasticPolicy, self).__init__()
        self._distribution = distribution

    def act(self, state, **kwargs):
        return self._distribution.sample_run(np.asarray(state)[np.newaxis, :])[0]


class GreedyStochasticPolicy(Policy):
    """
    returns action with the most probability in prob. distribution.
    """
    def __init__(self, distribution):
        """
        :param distribution:
        :type distribution NNDistribution
        """
        super(GreedyStochasticPolicy, self).__init__()
        self._distribution = distribution
        self._is_continuous = isinstance(distribution, NormalDistribution)

    def act(self, state, **kwargs):
        if self._is_continuous:
            return self._distribution.mean_run([state])[0]
        else:
            distribution = self._distribution.dist_run([state])[0]
            return np.argmax(distribution)


class KeepEpsilonPolicy(Policy):

    def __init__(self, n, n_distribution, q_function, epsilon, num_actions):
        super(KeepEpsilonPolicy, self).__init__(q_function, epsilon, num_actions)
        self._q_function, self._epsilon, self._num_actions = q_function, epsilon, num_actions
        self._n = n
        self._last_action, self._countdown = None, 0

    def act(self, state, exploration=True, **kwargs):
        """
        if exploration=False, do not keep previous action. True otherwise.
        :param state:
        :param exploration:
        :param kwargs:
        :return:
        """
        if not exploration:
            q_values = self._q_function(np.asarray(state)[np.newaxis, :])[0]
            action = np.argmax(q_values)
            return action

        if self._countdown > 0:
            self._countdown -= 1
        else:
            if np.random.rand() < self._epsilon:
                # random
                self._last_action = np.random.randint(self._num_actions)
            else:
                q_values = self._q_function(np.asarray(state)[np.newaxis, :])[0]
                self._last_action = np.argmax(q_values)
            self._countdown = self._n - 1
        return self._last_action

    def set_n(self, n):
        self._n = n
