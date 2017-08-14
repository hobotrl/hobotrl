# -*- coding: utf-8 -*-

import numpy as np
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

    def post_step(self, batch, info):
        if "score" in info:
            self._replay.update_score(batch["_index"], info["score"])


class TrajectoryOnSampler(Sampler):

    def __init__(self, replay_memory=None, interval=8, sample_maker=None):
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
        if replay_memory is None:
            replay_memory = playback.MapPlayback(interval, pop_policy="sequence")
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


class TruncateTrajectorySampler(Sampler):
    """
    sample {batch_size} trajectories of length {trajectory_length} in every {interval} steps.
    """
    def __init__(self, replay_memory=None, batch_size=8, trajectory_length=8, interval=4, sample_maker=None):
        """
        sample  trajectories from replay memory
        :param replay_memory:
        :type replay_memory: playback.Playback
        :param interval:
        :param sample_maker: callable to make samples
        """
        super(TruncateTrajectorySampler, self).__init__()
        if sample_maker is None:
            sample_maker = default_make_sample
        if replay_memory is None:
            replay_memory = playback.MapPlayback(1000)
        self._replay, self._sample_maker = replay_memory, sample_maker
        self._batch_size, self._trajectory_length, self._interval = \
            batch_size, trajectory_length, interval

        self._step_n = 0

    def step(self, state, action, reward, next_state, episode_done, **kwargs):
        """
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :param kwargs:
        :return: list of dict, each dict is a column-wise batch of transitions in a trajectory
        """
        self._step_n += 1
        self._replay.push_sample(self._sample_maker(state, action, reward, next_state, episode_done, **kwargs))
        if self._step_n % self._interval == 0 \
                and self._replay.get_count() > self._batch_size * self._trajectory_length * 4:
            batch_size = self._batch_size
            trajectories = []
            max_try = 8
            for i in range(max_try):
                batch = self._replay.sample_batch(batch_size)
                batch = playback.MapPlayback.to_rowwise(batch)
                traj = [self._trajectory_near(sample) for sample in batch]
                traj = filter(lambda x: x is not None, traj)
                trajectories.extend(traj)
                batch_size -= len(traj)
                if batch_size <= 0:
                    break
            if len(trajectories) < self._batch_size:
                # sample failed
                print "sample failed!"
                return None
            return trajectories
        else:
            return None

    def _trajectory_near(self, sample):
        """

        :param sample: row-wise sample
        :return: a truncated trajectory of length {trajectory_length} near sample.
        """
        push_index = self._replay.push_index
        capacity, count = self._replay.get_capacity(), self._replay.get_count()
        sample_i = sample["_index"]
        start, end, min_start, max_end = 0, 0, 0, 0
        if count < capacity:
            # replay not full
            max_end = count - 1
            min_start = 0
        else:
            # replay full
            max_end = push_index - 1 if push_index > sample_i else push_index + capacity
            min_start = push_index if push_index <= sample_i else push_index - capacity

        if max_end - min_start < self._trajectory_length:
            # not enough samples in replay memory
            return None
        # fetch x2 samples, and pick from it
        end = sample_i + self._trajectory_length
        end = max_end if end > max_end else end
        start = end - 2 * self._trajectory_length
        start = min_start if start < min_start else start
        nearby = self._get_batch(start, end)
        sample_i_nearby = sample_i - start
        episode_done = nearby["episode_done"]
        dones = np.argwhere(episode_done).flatten()
        for done_i_nearby in dones:
            if done_i_nearby < sample_i_nearby:
                min_start = sample_i - (sample_i_nearby - done_i_nearby)
            else:
                max_end = sample_i + (done_i_nearby - sample_i_nearby)
                # stop on first max_end
                break
        if max_end - min_start < self._trajectory_length:
            # not enough samples near sample_i, because of episode_done's
            return None
        nearby_row = playback.MapPlayback.to_rowwise(nearby)
        if max_end < sample_i + self._trajectory_length / 2:
            end = sample_i_nearby + (max_end - sample_i)
            result = nearby_row[end - self._trajectory_length + 1: end + 1]
        elif min_start > sample_i - self._trajectory_length /2 + 1 - self._trajectory_length % 2:
            start = sample_i_nearby - (sample_i - min_start)
            result = nearby_row[start: start + self._trajectory_length]
        else:
            end = sample_i_nearby + self._trajectory_length / 2
            result = nearby_row[end - self._trajectory_length + 1: end + 1]
        return playback.MapPlayback.to_columnwise(result)

    def _get_batch(self, start, end):
        return self._replay.get_batch((np.arange(start, end + 1) + self._replay.get_capacity())
                                      % self._replay.get_capacity())


class SamplerAgentMixin(object):
    def __init__(self, *args, **kwargs):
        super(SamplerAgentMixin, self).__init__(*args, **kwargs)
        pass


class TransitionBatchUpdate(SamplerAgentMixin):

    def __init__(self, sampler=None, *args, **kwargs):
        """
        :param sampler:
        :type sampler: TransitionSampler
        :param args:
        :param kwargs:
        """
        super(TransitionBatchUpdate, self).__init__(*args, **kwargs)
        if sampler is None:
            sampler = TransitionSampler(playback.MapPlayback(1000), 32, 4)
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


class TrajectoryBatchUpdate(SamplerAgentMixin):

    def __init__(self, sampler, *args, **kwargs):
        super(TrajectoryBatchUpdate, self).__init__(*args, **kwargs)
        self._sampler = sampler

    def reinforce_(self, state, action, reward, next_state, episode_done, **kwargs):
        super(TrajectoryBatchUpdate, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        batch = self._sampler.step(state, action, reward, next_state, episode_done, **kwargs)
        if batch is None:
            return {}
        info, sample_info = self.update_on_trajectory(batch)
        self._sampler.post_step(batch, sample_info)
        return info

    def update_on_trajectory(self, batch):
        """
        :param batch: list of column-wise batches, each batch guaranteed in trajectory order
        :return: (info, sample_info), in which `info` is a dict for invokers, and `sample_info` is a dict for sampler.post_step()
        """
        raise NotImplementedError()
