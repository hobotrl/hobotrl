# -*- coding: utf-8 -*-
"""
"""

from value_function import DeepQFuncActionOut
from hobotrl.mixin import BaseValueMixin

# TODO: unify action-in and action-out in this class?
class DeepQFuncMixin(BaseValueMixin):
    def __init__(self, **kwargs):
        super(DeepQFuncMixin, self).__init__(**kwargs)
        self.__dqf = DeepQFuncActionOut(**kwargs)
    
    def get_value(self, **kwargs):
        return self.__dqf.get_value(**kwargs)
        
    def improve_value_(self, *args, **kwargs):
        return self.__dqf.improve_value_(*args, **kwargs)
