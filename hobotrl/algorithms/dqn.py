# -*- coding: utf-8 -*-

from hobotrl.core import BaseAgent
from hobotrl.mixin import ReplayMixin, EpsilonGreedyPolicyMixin
from hobotrl.tf_dependent.mixin import DeepQFuncMixin

class DQN(
    ReplayMixin,
    EpsilonGreedyPolicyMixin,
    DeepQFuncMixin,
    BaseAgent):
    """
    """
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)

