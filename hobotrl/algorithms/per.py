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

    def __init__(self, importance_correction=1.0, **kwargs):
        """
        :param importance_correction: correction exponent term for importance sampling.
            could be a single float; or a callable for variant importance correction.
        :param kwargs:
        """
        super(PrioritizedExpReplayValue, self).__init__(**kwargs)
        self.importance_correction = importance_correction

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
            priority = batch["_priority"]
            sample_count = replay_buffer.get_count()
            print "priority, sample count:", priority, sample_count
            is_exponent = self.importance_correction() if callable(self.importance_correction) \
                else self.importance_correction
            w = np.power(sample_count * priority, -is_exponent)
            max_w = np.max(w)
            if max_w > 1.0:
                w = w / np.max(w)

            kwargs.update({"importance": w})
            print "importance:", w
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
                sample_index = batch["_index"]
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

