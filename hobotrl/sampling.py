# -*- coding: utf-8 -*-

import playback


def default_make_sample(state, action, reward, next_state, episode_done, **kwargs):
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "episode_done": episode_done
    }


class Sampler(object):
    """
    Sampler accepts transitions, possibly in trajectories, and returns sampled batch of data.
    """
    def step(self, state, action, reward, next_state, episode_done, **kwargs):
        """
        accept transitions, return sampled data when necessary
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :param kwargs:
        :return: None, or sampled data batch
        """
        raise NotImplementedError()

    def post_step(self, batch, info):
        pass


class TransitionSampler(Sampler):

    def __init__(self, replay_memory, batch_size, interval=1, minimum_count=None, sample_maker=None):
        """
        sample batch of transitions randomly from replay_memory.
        :param replay_memory:
        :type replay_memory: playback.Playback
        :param interval:
        :param sample_maker: callable to make samples
        """
        super(TransitionSampler, self).__init__()
        if sample_maker is None:
            sample_maker = default_make_sample
        if minimum_count is None:
            minimum_count = batch_size * 4
            if minimum_count > replay_memory.get_capacity():
                minimum_count = replay_memory.get_capacity()
        self._replay, self._sample_maker = replay_memory, sample_maker
        self._interval, self._batch_size, self._minimum_count = interval, batch_size, minimum_count
        self._step_n = 0

    def step(self, state, action, reward, next_state, episode_done, **kwargs):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :param kwargs:
        :return: dictionary, column-wise batch of i.i.d. transitions randomly sampled from replay memory
        """
        self._step_n += 1
        self._replay.push_sample(self._sample_maker(state, action, reward, next_state, episode_done, **kwargs))
        if self._step_n % self._interval == 0 and self._replay.get_count() >= self._minimum_count:
            return self._replay.sample_batch(self._batch_size)
        else:
            return None


class TrajectoryOnSampler(Sampler):

    def __init__(self, replay_memory, interval=8, sample_maker=None):
        """
        sample nearest trajectory or segment of trajectory for on-policy updates
        :param replay_memory:
        :type replay_memory: playback.Playback
        :param interval:
        :param sample_maker: callable to make samples
        """
        super(TrajectoryOnSampler, self).__init__()
        if sample_maker is None:
            sample_maker = default_make_sample
        self._replay, self._sample_maker = replay_memory, sample_maker
        self._interval = interval
        self._step_n = 0

    def step(self, state, action, reward, next_state, episode_done, **kwargs):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :param kwargs:
        :return: dictionary, column-wise batch of transitions of nearest trajectory
        """
        self._step_n += 1
        self._replay.push_sample(self._sample_maker(state, action, reward, next_state, episode_done, **kwargs))
        if self._step_n % self._interval == 0 or episode_done:
            batch = self._replay.sample_batch(self._replay.get_count())
            self._replay.reset()
            return batch
        else:
            return None


class SamplerAgentMixin(object):
    def __init__(self, *args, **kwargs):
        super(SamplerAgentMixin, self).__init__(*args, **kwargs)
        pass


class TransitionBatchUpdate(SamplerAgentMixin):

    def __init__(self, sampler, *args, **kwargs):
        super(TransitionBatchUpdate, self).__init__(*args, **kwargs)
        self._sampler = sampler

    def reinforce_(self, state, action, reward, next_state, episode_done, **kwargs):
        super(TransitionBatchUpdate, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        batch = self._sampler.step(state, action, reward, next_state, episode_done, **kwargs)
        if batch is None:
            return {}
        info, sample_info = self.update_on_transition(batch)
        self._sampler.post_step(batch, sample_info)
        return info

    def update_on_transition(self, batch):
        """
        :param batch:
        :return: (info, sample_info), in which `info` is a dict for invokers, and `sample_info` is a dict for sampler.post_step()
        """
        raise NotImplementedError()

