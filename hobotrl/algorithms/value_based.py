# -*- coding: utf-8 -*-


import tensorflow as tf

from hobotrl.core import Agent
from hobotrl.policy import EpsilonGreedyPolicy
import hobotrl.network as network
from hobotrl.tf_dependent.distribution import DiscreteDistribution


class ValueBasedAgent(Agent):

    def __init__(self, greedy_epsilon, num_actions, *args, **kwargs):
        kwargs.update({"greedy_epsilon": greedy_epsilon, "num_actions": num_actions})
        super(ValueBasedAgent, self).__init__(*args, **kwargs)
        self._q_function = self.init_value_function(*args, **kwargs)
        self._policy = self.init_policy(*args, **kwargs)

    def init_value_function(self, *args, **kwargs):
        """
        should be implemented by sub-classes.
        should return Q Function
        :param args:
        :param kwargs:
        :return:
        :rtype: network.Function
        """
        raise NotImplementedError()

    def init_policy(self, greedy_epsilon, num_actions, *args, **kwargs):
        return EpsilonGreedyPolicy(self._q_function, greedy_epsilon, num_actions)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)


class GreedyStateValueFunction(network.NetworkFunction):
    def __init__(self, q_function):
        """
        :param q_function: NetworkFunction calculating Q
        :type q_function: network.NetworkFunction
        """
        op_v = tf.reduce_max(q_function.output().op, axis=1)
        super(GreedyStateValueFunction, self).__init__(network.NetworkSymbol(op_v, "v", q_function.network),
                                                       q_function.inputs, q_function.variables)


class DoubleQValueFunction(network.NetworkFunction):
    def __init__(self, q_function, target_q_function):
        """
        :param q_function: NetworkFunction calculating Q
        :type q_function: network.NetworkFunction
        """
        self._q, self._target_q = q_function.output().op, target_q_function.output().op
        self._num_actions = self._q.shape.as_list()[-1]
        target_action = tf.argmax(self._q, axis=1)
        target_q_val = tf.reduce_sum(self._target_q * tf.one_hot(target_action, self._num_actions), axis=1)
        super(DoubleQValueFunction, self).__init__(network.NetworkSymbol(target_q_val, "v", q_function.network),
                                                       q_function.inputs, q_function.variables)


class StochasticStateValueFunction(network.NetworkFunction):
    def __init__(self, q_function, action_distribution):
        """
        :param q_function:
        :type q_function: network.NetworkFunction
        :param action_distribution:
        :type action_distribution: DiscreteDistribution
        """
        op_v = tf.reduce_sum(q_function.output().op * action_distribution.dist(), axis=1)

        super(StochasticStateValueFunction, self).__init__(
            network.NetworkSymbol(op_v, "v", q_function.network),
            q_function.inputs, q_function.variables)
