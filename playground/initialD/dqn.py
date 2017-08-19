# -*- coding: utf-8 -*-

from hobotrl.tf_dependent.base import BaseDeepAgent
import hobotrl as hrl
from hobotrl.core import BaseAgent
from hobotrl.mixin import ReplayMixin, EpsilonGreedyStickyPolicyMixin
from hobotrl.tf_dependent.mixin import DeepQFuncMixin


class DQNSticky(
    ReplayMixin,
    EpsilonGreedyStickyPolicyMixin,
    DeepQFuncMixin,
    BaseDeepAgent):
    """
    """
    def __init__(self, **kwargs):
        super(DQNSticky, self).__init__(**kwargs)

