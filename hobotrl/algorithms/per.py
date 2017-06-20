#
# -*- coding: utf-8 -*-
# Prioritized Exp Replay
# https://arxiv.org/pdf/1511.05952.pdf
#

import numpy as np
from hobotrl.tf_dependent.mixin import DeepQFuncMixin
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.mixin import ReplayMixin, EpsilonGreedyPolicyMixin
from hobotrl.playback import NearPrioritizedPlayback


class PrioritizedExpReplayValue(DeepQFuncMixin):

    def __init__(self, **kwargs):
        """
        """
        super(PrioritizedExpReplayValue, self).__init__(**kwargs)

    def improve_value_(self, state, action, reward, next_state,
                       episode_done, **kwargs):
        """Improve Q function with a random batch of expeirences.
        Ignore the single sample of experience passed in and use a batch of
        experiences randomly sampled from the replay buffer to update the
        Q function.
        """
        replay_buffer = self.get_replay_buffer()

        # if replay buffer has more samples than the batch_size.
        # TODO: the following is actually not necessary for sampling with replacement.
        batch_size = self._BATCH_SIZE
        if replay_buffer.get_count() >= batch_size:
            batch = replay_buffer.sample_batch(batch_size)
            batch = {k: np.array(v) for k, v in batch.iteritems()}  # force convert
            # check mandatory keys
            assert 'state' in batch
            assert 'action' in batch
            assert 'reward' in batch
            assert 'next_state' in batch
            # compute sampling weight
            importance_weight = batch.pop("_weight")
            sample_index = batch.pop("_index")
            batch["importance"] = importance_weight
            # sample `next_action` if not using greedy policy and the replay buffer
            # does not store `next_action` explicitly
            if not self._GREEDY_POLICY and 'next_action' not in batch:
                next_action = np.array(
                    [self.act(s_slice, **kwargs) for s_slice in batch['next_state']]
                )
                batch['next_action'] = next_action

            kwargs.update(batch)  # pass the batch in as kwargs
            kwargs.update({"sess": self.get_session()})

            info = self.get_qfunction().improve_value_(**kwargs)
            if "td_losses" in info:
                td_losses = info["td_losses"]
                replay_buffer.update_score(sample_index, td_losses)
            return info

        # if replay buffer is not filled yet.
        else:
            return {}


class PrioritizedDQN(
    ReplayMixin,
    EpsilonGreedyPolicyMixin,
    PrioritizedExpReplayValue,
    BaseDeepAgent):
    """
    """
    def __init__(self, **kwargs):
        kwargs["buffer_class"] = NearPrioritizedPlayback
        super(PrioritizedDQN, self).__init__(**kwargs)

