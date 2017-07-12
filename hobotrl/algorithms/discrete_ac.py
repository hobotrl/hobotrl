# -*- coding: utf-8 -*-

from hobotrl.mixin import ReplayMixin
from hobotrl.tf_dependent.mixin import DeepStochasticPolicyMixin, DeepQFuncMixin
from hobotrl.tf_dependent.base import BaseDeepAgent


class DiscreteActorCritic(
    DeepStochasticPolicyMixin,
    ReplayMixin,
    DeepQFuncMixin,
    BaseDeepAgent
):
    def __init__(self, **kwargs):
        super(DiscreteActorCritic, self).__init__(**kwargs)


