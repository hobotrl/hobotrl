#
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from value_function import DeepQFuncActionOut, DeepQFuncActionIn
from hobotrl.mixin import BaseValueMixin, BasePolicyMixin


class TFNetworkMixin(object):

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess

class DeepQFuncMixin(BaseValueMixin, TFNetworkMixin):
    """Mixin class for Q functions parameterized by deep neural networks.
    Initialize a parameterized action-value funciton as member.
    Dynamically insert the batch dimension before escalating the call
    for `get_value()`.
    Sample a batch of experience from the replay buffer in `improve_value_()`.

    NOTE: assumes there is a mixing class implementing the `get_replay_buffer()`
          method.
    """
    def __init__(self, is_action_in=False, **kwargs):
        super(DeepQFuncMixin, self).__init__(**kwargs)
        
        self.__IS_ACTION_IN = is_action_in

        if not is_action_in:
            self.__dqf = DeepQFuncActionOut(**kwargs)
        else:
            self.__dqf = DeepQFuncActionIn(**kwargs)

        self.__GREEDY_POLICY = False if is_action_in else self.__dqf.greedy_policy
        self.__BATCH_SIZE = kwargs['batch_size']

    def get_value(self, state, action=None, **kwargs):
        """Fetch action value(s)
        Wrapper for self.__dqf.get_value(). Checks and corrects arguments.
        """
        state, action = self.__check_shape(state, action)
        kwargs.update({"sess": self.sess})
        return self.__dqf.get_value(state, action, **kwargs)

    def improve_value_(self, state, action, reward, next_state,
                       episode_done, **kwargs):
        """Improve Q function with a random batch of expeirences.
        Ignore the single sample of experience passed in and use a batch of
        experiences randomly sampled from the replay buffer to update the
        Q function.
        """
        replay_buffer = self.get_replay_buffer()

        # if replay buffer has more samples than the batch_size.
        # TODO: the following is actually not necessary for sampling with replaycement.
        if replay_buffer.get_count() >= self.__BATCH_SIZE:
            batch = replay_buffer.sample_batch(self.__BATCH_SIZE)
            batch = {k: np.array(v) for k, v in batch.iteritems()}  # force convert

            # check mandatory keys
            assert 'state' in batch
            assert 'action' in batch
            assert 'reward' in batch
            assert 'next_state' in batch
            
            # sample `next_action` if not using greedy policy and the replay buffer
            # does not store `next_action` explicitly
            if not self.__GREEDY_POLICY and 'next_action' not in batch:
                next_action = np.array(
                    [self.act(s_slice, **kwargs) for s_slice in batch['next_state']]
                )
                batch['next_action'] = next_action

            kwargs.update(batch)  # pass the batch in as kwargs
            kwargs.update({"sess": self.sess})
            return self.__dqf.improve_value_(**kwargs)

        # if replay buffer is not filled yet.
        else:
            info_key = 'DeepQFuncMixin\\debug_str'
            return {
                info_key: 'replay buffer not filled yet.'
            }
      
    def get_grad_q_action(self, state, action=None, **kwargs):
        """Fetch action value(s)
        Wrapper for self.__dqf.get_grad_q_action(). Checks and corrects
        arguments. Raise exception for action-out network.
        """
        if self.__IS_ACTION_IN:
            state, action = self.__check_shape(state, action)
            kwargs.update({"sess": self.sess})
            return self.__dqf.get_grad_q_action(state, action, **kwargs)
        else:
            raise NotImplementedError(
                "DeepQFuncMixin.get_grad_q_action(): "
                "dQ/da is not defined for action-out network."
            )
    
    @property
    def deep_q_func(self):
        return self.__dqf

    def __check_shape(self, state, action):
        """Shape checking procedure
        Convert both state and action to numpy arrays. Prepends the batch
        dimension whereever needed and make sure the batch sizes of state
        and action are equal.
        """
        # force convert to numpy array
        state = np.array(state)
        action = np.array(action) if action is not None else None

        # assert that action must be explicitly provided for action-in Q func.
        if self.__IS_ACTION_IN:
            assert action is not None

        # === Insert the batch dimension whenever needed ===
        if state.shape == self.__dqf.state_shape:  # non-batch single sample
            # prepend the batch dimension for state (and action if provided)
            state = state[np.newaxis, :]
            if action is not None:
                action = action[np.newaxis]
                # assert action is a vector 
                assert len(action.shape) == 1
        else:  # batch format
            if action is not None:
                # squeeze out redundant dims in action (except the batch dim)
                sqz_dims = [i for i, n in enumerate(action.shape) if i>0 and n==1]
                action = np.squeeze(action, axis=sqz_dims)
                # assert action is a vector and batch size match with state
                assert len(action.shape)==1 and state.shape[0]==action.shape[0]

        return state, action


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
