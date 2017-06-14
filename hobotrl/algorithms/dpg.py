# -*- coding: utf-8 -*-
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.mixin import ReplayMixin, OUExplorationMixin
from hobotrl.tf_dependent.mixin import DeepQFuncMixin, DeepDeterministicPolicyMixin


class DPG(
    ReplayMixin,
    OUExplorationMixin,
    DeepDeterministicPolicyMixin,
    DeepQFuncMixin,
    BaseDeepAgent
):
    def __init__(self, **kwargs):
        kwargs['is_action_in'] = True
        super(DPG, self).__init__(**kwargs)

