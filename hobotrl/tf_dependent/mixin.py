#
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from value_function import DeepQFuncActionOut
from hobotrl.mixin import BaseValueMixin, BasePolicyMixin


class TFNetworkMixin(object):

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess


# TODO: unify action-in and action-out in this class?
class DeepQFuncMixin(BaseValueMixin, TFNetworkMixin):
    """Mixin class for DNN parameterized Q functions.
    Initialize a parameterized action-value funciton as member.
    Dynamically insert the batch dimension before escalating the call
    for `get_value()`.
    Sample a batch of experience from the replay buffer in `improve_value_()`.

    NOTE: assumes there is a mixing class implementing the `get_replay_buffer()`
          method.
    """
    def __init__(self, **kwargs):
        super(DeepQFuncMixin, self).__init__(**kwargs)
        self.__dqf = DeepQFuncActionOut(**kwargs)
        self.batch_size = kwargs['batch_size']

    def get_value(self, state, action=None, **kwargs):
        """Fetch action value(s)
        Wrapper for self.__dqf.get_value(). Prepends the batch dimension
        whenever needed and make sure the batch sizes of `state` and
        `action` are equal.

        Parameters:
        -----------
        state :
        action :
        """
        dqf_state_shape = self.__dqf.state_shape
        state = np.array(state)  # force convert to np.array

        # Insert the batch dimension whenever needed
        if state.shape == dqf_state_shape:  # non-batch format
            state = state[np.newaxis, :]  # insert batch dimension
            if action is not None:
                action = np.array(action)[np.newaxis]
                assert len(action.shape)==1  # assert action is a vector 
        else:  # batch format
            assert state.shape[1:] == dqf_state_shape
            if action is not None:
                # squeeze out redundant dims to make a vector
                sqz_dims = [
                    i for i, n in enumerate(action.shape) if i>0 and n==1
                ]  # we want to keep the batch dimension
                action = np.squeeze(action)
                # assert action is a vector and batch size match
                assert len(action.shape)==1 and state.shape[0]==action.shape[0]

        kwargs.update({"sess": self.sess})
        return self.__dqf.get_value(state, action, **kwargs)

    def improve_value_(self, state, action, reward, next_state, episode_done, **kwargs):
        """Improve Q function with batch of uncorrelated expeirences.
        Ignore the piece of experience passed in and use a batch of experiences
        sampled from the replay buffer to update the value function.
        """
        replay_buffer = self.get_replay_buffer()

        # if replay buffer has more samples than the batch_size.
        if replay_buffer.get_count() >= self.batch_size:
            batch = replay_buffer.sample_batch(self.batch_size)
            for k, v in batch.iteritems():
                batch[k] = np.array(v)

            # check mandatory keys
            assert 'state' in batch
            assert 'action' in batch
            assert 'reward' in batch
            assert 'next_state' in batch

            # call the actual member method
            kwargs.update(batch)  # pass the batch in as kwargs
            kwargs.update({"sess": self.sess})
            return self.__dqf.improve_value_(**kwargs)

        # if replay buffer is not filled yet.
        else:
            info_key = 'DeepQFuncMixin\\debug_str'
            return {
                info_key: 'replay buffer not filled yet.'
            }


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
