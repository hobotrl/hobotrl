# -*- coding: utf-8 -*-

from hobotrl.mixin import ReplayMixin
from hobotrl.tf_dependent.mixin import DeepStochasticPolicyMixin, DeepQFuncMixin
from hobotrl.tf_dependent.base import BaseDeepAgent


class ActorCritic(
    DeepStochasticPolicyMixin,
    ReplayMixin,
    DeepQFuncMixin,
    BaseDeepAgent
):
    def __init__(self, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)


