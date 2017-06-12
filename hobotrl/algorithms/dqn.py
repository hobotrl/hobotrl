# -*- coding: utf-8 -*-

from hobotrl.tf_dependent.base import BaseDeepAgent
import hobotrl as hrl
from hobotrl.core import BaseAgent
from hobotrl.mixin import ReplayMixin, EpsilonGreedyPolicyMixin
from hobotrl.tf_dependent.mixin import DeepQFuncMixin


class DQN(
    ReplayMixin,
    EpsilonGreedyPolicyMixin,
    DeepQFuncMixin,
    BaseDeepAgent):
    """
    """
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)

