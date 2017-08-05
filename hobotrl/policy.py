# -*- coding: utf-8 -*-


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
        q_values = self.q_function(np.asarray([state]))[0]
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

    def act(self, state, **kwargs):
        if np.random.rand() < self._epsilon:
            # random
            return np.random.randint(self._num_actions)
        q_values = self.q_function(np.asarray([state]))[0]
        action = np.argmax(q_values)
        return action


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
        return self._distribution.sample_run(np.asarray([state]))[0]


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