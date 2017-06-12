# -*- coding: utf-8 -*-
"""Mixin classes for Tensorflow-based RL modules
"""

import numpy as np
import hobotrl as hrl
from value_function import DeepQFuncActionOut
from hobotrl.mixin import BaseValueMixin, BasePolicyMixin
from value_function import DeepQFuncActionOut, DeepQFuncActionIn
from policy import DeepDeterministicPolicy
from policy import NNStochasticPolicy


class DeepQFuncMixin(BaseValueMixin):
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
            return {}
      
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
        if list(state.shape) == list(self.__dqf.state_shape):  # non-batch single sample
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
                # action = np.squeeze(action, axis=sqz_dims)
                # assert action is a vector and batch size match with state
                # assert len(action.shape)==1 and state.shape[0]==action.shape[0]
                assert state.shape[0] == action.shape[0]

        return state, action


class NNStochasticPolicyMixin(BasePolicyMixin):

    def __init__(self, **kwargs):
        kwargs.update({
            "parent_agent": self
        })
        self._policy = NNStochasticPolicy(**kwargs)
        super(NNStochasticPolicyMixin, self).__init__(**kwargs)

    def act(self, state, **kwargs):
        kwargs.update({
            "sess": self.get_session()
        })
        return self._policy.act(state, **kwargs)[0]

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        info = super(NNStochasticPolicyMixin, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        kwargs.update({
            "sess": self.get_session()
        })
        info.update(self._policy.update_policy(state, action, reward, next_state, episode_done, **kwargs))
        return info


# TODO: inherit from a base class which improves policy?
class DeepDeterministicPolicyMixin(BasePolicyMixin):
    def __init__(self, **kwargs):
        super(DeepDeterministicPolicyMixin, self).__init__(**kwargs)
        self.__ddp = DeepDeterministicPolicy(**kwargs)
        self.__BATCH_SIZE = kwargs['batch_size']

    def act(self, state, **kwargs):
        state = np.array(state)
        assert state.shape == self.__ddp.state_shape
        state = state[np.newaxis, :]
        kwargs.update({"sess": self.sess})
        return self.__ddp.act(state=state, **kwargs)

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        parent_info = super(DeepDeterministicPolicyMixin, self).reinforce_(
            state=state, action=action, reward=reward, next_state=next_state,
            episode_done=episode_done, **kwargs
        )
        self_info = self.improve_policy_(
            state, action, reward, next_state, episode_done, **kwargs
        )
        return parent_info.update(self.info)

    def improve_policy_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):

        replay_buffer = self.get_replay_buffer()

        # if replay buffer has more samples than the batch_size.
        if replay_buffer.get_count() >= self.__BATCH_SIZE:
            batch = replay_buffer.sample_batch(self.__BATCH_SIZE)
            batch = {k: np.array(v) for k, v in batch.iteritems()}  # force convert

            # check mandatory keys
            assert 'state' in batch
            assert 'action' in batch
            
            # get value gradient from value func
            batch['grad_q_action'] = self.get_grad_q_action(
                batch['state'], batch['action']
            )
            
            kwargs.update(batch)  # pass the batch in as kwargs
            kwargs.update({"sess": self.sess})
            return self.__dqf.improve_policy_(**kwargs)
        # if replay buffer is not filled yet.
        else:
            return {}
