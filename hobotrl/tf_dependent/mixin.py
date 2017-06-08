#
# -*- coding: utf-8 -*-
"""
"""

from value_function import DeepQFuncActionOut
from hobotrl.mixin import BaseValueMixin, BasePolicyMixin


class TFNetworkMixin(object):

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess


# TODO: unify action-in and action-out in this class?
class DeepQFuncMixin(BaseValueMixin, TFNetworkMixin):
    def __init__(self, **kwargs):
        super(DeepQFuncMixin, self).__init__(**kwargs)
        self.__dqf = DeepQFuncActionOut(**kwargs)
        self.batch_size = kwargs['batch_size']
    
    def get_value(self, **kwargs):
        kwargs.update({
            "sess": self.sess
        })
        return self.__dqf.get_value(**kwargs)
        
    def improve_value_(self, state, action, reward, next_state, episode_done, **kwargs):
        replay_buffer = self.get_replay_buffer()
        if replay_buffer.get_count() >= self.batch_size:
            batch = replay_buffer.sample_batch(self.batch_size)
            kwargs.update(batch)
            kwargs.update({
                "sess": self.sess
            })
            return self.__dqf.improve_value_(**kwargs)
        else:
            return {}


class NNStochasticPolicyMixin(BasePolicyMixin, TFNetworkMixin):
    def __init__(self, **kwargs):
        super(NNStochasticPolicyMixin, self).__init__(**kwargs)

    def act(self, state, **kwargs):
        raise NotImplementedError()

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        info = super(NNStochasticPolicyMixin, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        info.update(self.update_policy(state, action, reward, next_state, episode_done, **kwargs))
        return info

    def update_policy(self, state, action, reward, next_state, episode_done, **kwargs):
        raise NotImplementedError()
