# -*- coding: utf-8 -*-
"""Mixin classes for Tensorflow-based RL modules
"""

import numpy as np
import hobotrl as hrl
from hobotrl.mixin import BaseValueMixin, BasePolicyMixin
from hobotrl.playback import MapPlayback
from value_function import DeepQFuncActionOut, DeepQFuncActionIn
from policy import NNStochasticPolicy, DeepDeterministicPolicy


class DeepQFuncMixin(BaseValueMixin):
    """Mixin class for Q functions parameterized by deep neural networks.
    Initialize a parameterized action-value funciton as member.
    Dynamically insert the batch dimension before escalating the call
    for `get_value()`.
    Sample a batch of experience from the replay buffer in `improve_value_()`.

    NOTE: assumes there is a mixing class implementing the `get_replay_buffer()`
          method.
    """
    def __init__(self, dqn_param_dict, is_action_in=False, **kwargs):
        super(DeepQFuncMixin, self).__init__(**kwargs)

        self.__IS_ACTION_IN = is_action_in
        dqn_param_dict.update(kwargs)
        if not is_action_in:
            self.__dqf = DeepQFuncActionOut(**dqn_param_dict)
        else:
            self.__dqf = DeepQFuncActionIn(**dqn_param_dict)

        # for action-in, greedy_policy will by pass exploration for act() 
        # TODO: this means the no-exploration limit of the policy is greedy.
        #       while this is true for DPG. Does this generally holds
        #       sementically? Or maybe on- and off-policy is more semantically
        #       accurate.
        self.__GREEDY_POLICY = self.__dqf.greedy_policy
        self.__BATCH_SIZE = kwargs['batch_size']

    def get_value(self, state, action=None, **kwargs):
        """Fetch action value(s)
        Wrapper for self.__dqf.get_value(). Checks and corrects arguments.
        """
        state, action, is_batch = self.__check_shape(state, action)
        kwargs.update({"sess": self.sess})
        return self.__dqf.get_value(state, action, is_batch=is_batch, **kwargs)

    def get_qfunction(self):
        return self.__dqf

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
            next_state = batch['next_state']
            if 'next_action' not in batch:
                if not self.__GREEDY_POLICY:  # policy + exploration
                    next_action = self.act(
                        next_state, exploration_off=False, use_target=True,
                        batch=True, **kwargs
                    )
                else:  # pure policy, no exploration
                    next_action = self.act(
                        next_state, exploration_off=True, use_target=True,
                        batch=True, **kwargs
                    )
                batch['next_action'] = np.array(next_action)

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
            state, action, is_batch = self.__check_shape(state, action)
            kwargs.update({"sess": self.sess})
            return self.__dqf.get_grad_q_action(state, action, is_batch, **kwargs)
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
        and action are equal. Also returns an indicator for batch case.

        :param state: state.
        :param action: action.
        :return state: converted state.
        :return action: converted action.
        :return is_batch: indicator for the batch case.
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
                assert len(action.shape) == 1
            is_batch = False
        else:  # batch format
            if action is not None:
                assert state.shape[0] == action.shape[0]
            is_batch = True
        return state, action, is_batch


class NNStochasticPolicyMixin(BasePolicyMixin):

    def __init__(self, nnsp_param_dict, update_method, update_interval, **kwargs):
        """Initialization

        :param nnsp_param_dict: kwarg dict for nnsp Initialization.
        """
        super(NNStochasticPolicyMixin, self).__init__(**kwargs)
        self.__nnsp = NNStochasticPolicy(**nnsp_param_dict)
        self.__BATCH_SIZE = kwargs['batch_size']  # assume exist for init. replay_buffer
        # assign proper update method
        dict_update_methods = {
            "multistep": self.update_multistep_,
            "replayed": self.update_replayed_,
        }
        self.__update_method = dict_update_methods[update_method]
        self.__countdown_update = update_interval
        self.__UPDATE_INTERVAL = update_interval
        # initialize a buffer to store episodic experiences
        if update_method=='episodic':
            self.episode_buffer = MapPlayback(
                update_interval,
                {"state": state_shape,
                 "action": action_shape,
                 "reward": (),
                 "episode_done": (),
                 "next_state": state_shape,
                 "state_value": (),},
                pop_policy="sequence"
            )

    def act(self, state, batch=False, **kwargs):
        """Emit action for this state.
        Accepts both a single sample and a batch of samples. In the former
        case, automatically insert the batch dimension to match the shape of
        placeholders and use determistic inference to avoid inconsistency with
        training phase. For the latter case, simply pass along the arguments to
        the ddp member. Assumes there is a mixing class implementing the
        `get_replay_buffer()` method.

         TODO: do not use 'batch', distinguish batch case using
              state.shape

        :param state: the state.
        """
        state = np.array(state)
        # prepend batch dim and use deterministic inference for single sample
        if not batch:
            assert list(state.shape) == list(self.__nnsp.state_shape)
            state = state[np.newaxis, :]
            # TODO: nnsp currently doesn't support `is_training` arg.
            return self.__nnsp.act(state, is_training=False, **kwargs)[0, :]
        # use default stochastic inference of batch
        else:
            return self.__nnsp.act(state, **kwargs)

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        parent_info = super(NNStochasticPolicyMixin, self).reinforce_(
            state=state, action=action, reward=reward, next_state=next_state,
            episode_done=episode_done, **kwargs)
        self_info = self.improve_policy_(
            state, action, reward, next_state, episode_done, **kwargs
        )
        parent_info.update(self_info)
        return parent_info

    def improve_policy_(self, state, action, reward, next_state,
                        episode_done=False, **kwargs):
        """Wraps around the desired update method."""
        return self.__update_method(
            state, action, reward, next_state, episode_done, **kwargs
        )

    def update_direct_():
        pass

    def update_multistep_(self, state, action, reward, next_state,
                         episode_done=False, **kwargs):
        """
        Update policy using episodic reward
        """
        # record new sample
        state_value = self.get_state_value(state=state, **kwargs)
        self.episode_buffer.push_sample(
            sample={
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': np.asarray([reward], dtype=float),
                'episode_done': np.asarray([episode_done], dtype=float),
                'state_value': np.asarray([state_value], dtype=float)
            }
        )
        # empty buffer and update
        self.__countdown_update -= 1
        if episode_done or self.__countdown_update == 0:
            self.__countdown_update = self.__UPDATE_INTERVAL
            # MapPlayback with 'sequence' option pop out trajectory
            len_path = self.episode_buffer.get_count()
            path = self.episode_buffer.sample_batch(len_path)
            self.episode_buffer.reset()
            # unpack
            state = np.asarray(trajectory['state'])
            action = trajectory['action']
            reward = trajectory['reward']
            next_state = np.asarray(trajectory['next_state'])
            episode_done = trajectory['episode_done']
            state_value = trajectory['state_value']
            # TODO: is explicitly setting r to 0.0 necessary. Assuming Q that is
            # estimated correctly, it should include such information.
            G = np.zeros(shape=[batch_size], dtype=float)  # front return
            r = self.get_value(
                    state=np.asarray([next_state]),
                    is_batch=False, **kwargs
                )[0]  # tail return
            for i in range(batch_size):
                index = batch_size -1 - i
                r = reward[index] + self.reward_decay * r
                G[index] = r
            advantage = G - state_value

    def update_replayed_():
        pass

class DeepDeterministicPolicyMixin(BasePolicyMixin):
    """Wrapper mixin for a DDP."""
    def __init__(self, ddp_param_dict, **kwargs):
        """Initialization.

        :param ddp_param_dict: kwarg dict for ddp init.
        """
        super(DeepDeterministicPolicyMixin, self).__init__(**kwargs)
        self.__ddp = DeepDeterministicPolicy(**ddp_param_dict)
        self.__BATCH_SIZE = kwargs['batch_size']  # for sampling replay buffer

    def act(self, state, batch=False, **kwargs):
        """Emit action for this state.
        Accepts both a single sample and a batch of samples. In the former
        case, automatically insert the batch dimension to match the shape of
        placeholders and use determistic inference to avoid inconsistency with
        training phase. For the latter case, simply pass along the arguments to
        the ddp member. Assumes there is a mixing class implementing the
        `get_replay_buffer()` method.

        :param state: the state.
        """
        state = np.array(state)
        # prepend batch dim and use deterministic inference for single sample
        if not batch:
            assert list(state.shape) == list(self.__ddp.state_shape)
            state = state[np.newaxis, :]
            return self.__ddp.act(state, is_training=False, **kwargs)[0, :]
        # use default stochastic inference of batch
        else:
            return self.__ddp.act(state, **kwargs)

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        parent_info = super(DeepDeterministicPolicyMixin, self).reinforce_(
            state=state, action=action, reward=reward, next_state=next_state,
            episode_done=episode_done, **kwargs
        )
        self_info = self.improve_policy_(
            state, action, reward, next_state, episode_done, **kwargs
        )
        parent_info.update(self_info)
        return parent_info

    def improve_policy_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        """Improve policy network with samples from the replay_buffer."""
        replay_buffer = self.get_replay_buffer()
        # if replay buffer has more samples than the batch_size.
        if replay_buffer.get_count() >= self.__BATCH_SIZE:
            batch = replay_buffer.sample_batch(self.__BATCH_SIZE)
            state = np.array(batch['state'])
            kwargs.update({'sess': self.sess})
            # TODO: here we need two forward passes for the policy network:
            #       One time for accesing policy, the other for computing
            #       DPG. Can we do better?
            # TODO: the following forward-backward pass are executed no matter
            #       what. But ddp only needs it when countdown touches zero.
            #       Maybe providing a function handle is better? But it is
            #       messy to manage those handles anyway....
            action_on = self.act(
                state, exploration_off=True, use_target=False, batch=True,
                **kwargs
            )
            grad_q_action = self.get_grad_q_action(state, action_on)
            info = self.__ddp.improve_policy_(
                state, grad_q_action, **kwargs
            )
            return info
        # if replay buffer is not filled yet.
        else:
            return {}


