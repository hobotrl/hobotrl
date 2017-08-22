# -*- coding: utf-8 -*-


import tensorflow as tf

from hobotrl.core import Agent
from hobotrl.policy import EpsilonGreedyPolicy
import hobotrl.network as network


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


class StateValueFunction(network.NetworkFunction):
    def __init__(self, q_function):
        """
        :param q_function: NetworkFunction calculating Q
        :type q_function: network.NetworkFunction
        """
        op_v = tf.reduce_max(q_function.output().op, axis=1)
        super(StateValueFunction, self).__init__(network.NetworkSymbol(op_v, "v", q_function.network),
                                                 q_function.inputs, q_function.variables)


