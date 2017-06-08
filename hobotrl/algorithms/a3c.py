#
# -*- coding: utf-8 -*-

import hobotrl as hrl
import hobotrl.tf_dependent as tf_deps


class DiscreteA3C(
    tf_deps.mixin.DeepQFuncMixin,
    hrl.core.BaseAgent
):

    def act(self, state, evaluate=False, **kwargs):
        pass

    def __init__(self, **kwargs):
        super(DiscreteA3C, self).__init__(**kwargs)
