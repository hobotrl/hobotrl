# -*- coding: utf-8 -*-
"""Mixin classes for Tensorflow-based RL modules
"""

import numpy as np
from scipy.linalg import toeplitz

import hobotrl as hrl
from hobotrl.mixin import BaseValueMixin, BasePolicyMixin
from hobotrl.playback import MapPlayback
from value_function import DeepQFuncActionOut, DeepQFuncActionIn
from policy import DeepStochasticPolicy, DeepDeterministicPolicy


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
        kwargs["dqn_param_dict"] = dqn_param_dict
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
        self._GREEDY_POLICY = self.__dqf.greedy_policy
        self._BATCH_SIZE = kwargs['batch_size']

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
        if replay_buffer.get_count() >= self._BATCH_SIZE:
            batch = replay_buffer.sample_batch(self._BATCH_SIZE)
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
                if not self._GREEDY_POLICY:  # policy + exploration
                    kwargs.update({
                        'exploration_off': False,
                        'use_target': True,
                        'batch': True})
                    next_action = self.act(next_state, **kwargs)
                else:  # pure policy, no exploration
                    kwargs.update({
                        'exploration_off': True,
                        'use_target': True,
                        'batch': True})
                    next_action = self.act(next_state, **kwargs)
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
            if "sess" not in kwargs:
                kwargs.update({"sess": self.sess})
            return self.__dqf.get_grad_q_action(state, action, is_batch=is_batch, **kwargs)
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


class DeepStochasticPolicyMixin(BasePolicyMixin):
    """Wrapper mixin for a DSP."""
    def  __init__(self, dsp_param_dict, backup_method, update_interval, gamma,
                  backup_depth=None, **kwargs):
        """Initialization

        :param dsp_param_dict: kwarg dict for dsp Initialization.
        :param backup_method: backup method for calculating eligibility.
        :param update_interval: periodicity of SPG update.
        :param gamma: reward discount factor.
        :param backup_depth: length of path used for back up eligibility values.
        """
        super(DeepStochasticPolicyMixin, self).__init__(**kwargs)
        self.__dsp = DeepStochasticPolicy(**dsp_param_dict)
        self.__BATCH_SIZE = kwargs['batch_size']  # assume exist for init. replay_buffer
        self.is_continuous_action = self.__dsp.is_continuous_action
        state_shape, action_shape = self.__dsp.state_shape, self.__dsp.action_shape
        self.reward_decay = gamma
        # assign proper update method
        dict_backup_method = {
            "multistep": self.__backup_multistep,  # multi-step backup along a path
            "replayed": self.__backup_replayed,  # replayed one-step backup
        }
        self.__fun_backup = dict_backup_method[backup_method]
        self.__countdown_update = update_interval
        self.__UPDATE_INTERVAL = update_interval
        self.__BACKUP_DEPTH = backup_depth if backup_depth is not None \
            else update_interval
        # initialize a buffer to store episodic experiences
        if backup_method=='multistep':
            self.episode_buffer = MapPlayback(
                self.__BACKUP_DEPTH,
                {"state": state_shape,
                 "action": action_shape,
                 "reward": (),
                 "episode_done": (),
                 "next_state": state_shape},
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
            assert list(state.shape) == list(self.__dsp.state_shape)
            state = state[np.newaxis, :]
            # TODO: dsp currently doesn't support `is_training` arg.
            return self.__dsp.act(state, is_training=False, **kwargs)[0]
        # use default stochastic inference of batch
        else:
            return self.__dsp.act(state, **kwargs)

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        parent_info = super(DeepStochasticPolicyMixin, self).reinforce_(
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
        return self.__fun_backup(
            state, action, reward, next_state, episode_done, **kwargs
        )

    def get_state_value_(self, state, **kwargs):
        """Calculate state value based on action value.

        This method is called to calculate the state value when the value mixin
        represents an action value function.

        For the continuous action case, assume local linearity for the action
        value function and use mean action to approximate the state value fcn.

        For the discrete action case, use the action distribution from current
        policy to average action values to get the state value.
        """
        # Retrieve tf session
        assert 'sess' in kwargs
        sess = kwargs['sess']
        # Average action values to get the state value
        if self.is_continuous_action:
            # Continuous action case:
            # V(s)=\integral_a{\pi(s, a)*Q(s, a)}. Here we use the action value
            # of mean action to avoid the integral. Note this is only
            # approximately true when Q(s,a) is linear in `a'.
            action_mean = self.__dsp.distribution.mean_run(sess, [state])
            V = self.get_value(state=state, action=action_mean, **kwargs)
        else:
            # Discrete action case:
            # V(s) = \sum_a{\pi(s,a)*Q(s,a)}.
            Q = self.get_value(state=state)
            dist = self.__dsp.distribution.dist_run(
                sess=sess, inputs=[state],
            )
            V = np.sum(Q*dist, axis=1)
        return V

    def __backup_multistep(self, state, action, reward, next_state,
                           episode_done=False, **kwargs):
        """Update policy using return calculated from path rewards."""
        # record new sample in path
        self.episode_buffer.push_sample(
            sample={
                'state': np.array(state),
                'action': np.array(action),
                'next_state': np.array(next_state),
                'reward': np.asarray(reward, dtype=float),
                'episode_done': np.asarray(episode_done, dtype=float),
            }
        )
        self.__countdown_update -= 1
        # pop out path experience for pg
        if episode_done or self.__countdown_update == 0:
            self.__countdown_update = self.__UPDATE_INTERVAL
            # pop out path up to now
            len_path = self.episode_buffer.get_count()
            path = self.episode_buffer.sample_batch(len_path)
            # index of the last sample in path
            tail_index = (self.episode_buffer.data['state'].push_index - 1) % \
                self.episode_buffer.capacity
            # only reset buffer when episode ends, otherwise
            # use as a circular buffer
            if episode_done:
                self.episode_buffer.reset()
            # unpack
            state = np.asarray(path['state'])
            action = path['action']
            reward = path['reward']
            next_state = np.asarray(path['next_state'])
            episode_done = path['episode_done']
            # calculate return
            gamma = self.reward_decay
            tail_return = self.get_value(state=next_state[tail_index], **kwargs)[0]
            total_return = self.__path_return(
                reward, gamma, tail_return, tail_index
            )
            state_value = self.get_state_value_(state=state, **kwargs)
            advantage = total_return - state_value
            return self.__dsp.improve_policy_(
                state=state, action=action, advantage=advantage, **kwargs
            )
        else:
            return {}

    def __backup_replayed(self, state, action, reward, next_state,
                         episode_done=False, **kwargs):
        raise NotImplementedError()

    def __path_return(self, reward, gamma, tail_return=0.0, tail_index=-1):
        len_path = len(reward)
        G = np.zeros(len_path)
        for i in range(len_path):
            idx = (tail_index - i) % len_path
            tail_return = reward[idx] + gamma*tail_return
            G[idx] = tail_return
        return G


class DeepDeterministicPolicyMixin(BasePolicyMixin):
    """Wrapper mixin for a DDP."""
    def __init__(self, ddp_param_dict, **kwargs):
        """Initialization.

        :param ddp_param_dict: kwarg dict for ddp init.
        """
        kwargs['ddp_param_dict'] = ddp_param_dict
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


