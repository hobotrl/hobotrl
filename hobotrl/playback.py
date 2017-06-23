#
# -*- coding: utf-8 -*-

import logging
import numpy as np


class Playback(object):
    def __init__(self, capacity, sample_shape, push_policy="sequence", pop_policy="random",
                 augment_offset=None, augment_scale=None, dtype=None):
        """
        stores ndarray.
        :param capacity: total count of samples stored
        :param sample_length: length of a single sample
        :param push_policy: sequence
        :param pop_policy: sequence/random
        :param augment_offset:
        :param augment_scale:
                sample apply transformation: (sample + offset) * scale before returned by sample_batch()
        :param dtype: np.float32 default
        """
        self.capacity = capacity
        self.sample_shape = list(sample_shape)
        print "capacity:", capacity, ", sample_len:", sample_shape
        # self.data = np.ndarray(shape=([self.capacity] + self.sample_shape),
        #                        dtype=dtype)
        self.data = None
        self.push_policy = push_policy
        self.pop_policy = pop_policy

        self.augment_offset = 0 if augment_offset is None else augment_offset
        self.augment_scale = 1 if augment_scale is None else augment_scale
        self.count = 0
        self.push_index = 0
        self.pop_index = 0

    def get_count(self):
        """
        current count of samples stored
        :return:
        """
        return self.count

    def get_capacity(self):
        """
        maximum count of samples can be stored
        :return:
        """
        return self.capacity

    def reset(self):
        """
        clear all samples
        :return:
        """
        self.count, self.push_index, self.pop_index = 0, 0, 0

    def add_sample(self, sample, index, sample_score=0):
        """
        add sample to specified position
        :param sample:
        :param index:
        :param sample_score:
        :return:
        """
        if self.data is None:
            # lazy creation
            print "initializnig data with:", sample
            self.data = np.zeros(shape=([self.capacity] + self.sample_shape),
                                   dtype=sample.dtype)
            print "initializing data:", self.data.shape, ",", self.data.dtype
        self.data[index] = sample
        if self.count < self.capacity:
            self.count += 1

    def push_sample(self, sample, sample_score=0):
        """
        add sample into playback
        :param sample:
        :param sample_score:
        :return:
        """
        self.add_sample(sample, self.push_index, sample_score)
        self.push_index = (self.push_index + 1) % self.capacity

    def next_batch_index(self, batch_size):
        """
        calculate index of next batch
        :param batch_size:
        :return:
        """
        if self.get_count() == 0:
            return np.asarray([], dtype=int)
        if self.pop_policy == "random":
            index = np.random.randint(0, self.count, batch_size)
        elif self.pop_policy == "sequence":
            '''
            pop batch as [0, step, 2 * step, ..., n - step],
                [1, step + 1, 2 * step + 1, ..., n - step + 1],
                [2, step + 2, 2 * step + 2, ..., n - step + 2],
                ...,
                [step - 1, 2 * step - 1, ..., n - 1]
            '''
            step = (self.count / batch_size)
            step = step + 1 if self.count % batch_size != 0 else step
            index = self.pop_index % step
            index = np.arange(index, self.count, step)
            self.pop_index = (self.pop_index + 1) % step
        return index

    def get_batch(self, index):
        """
        get batch by index
        :param index:
        :return:
        """
        return (self.data[index] + self.augment_offset) * self.augment_scale

    def sample_batch(self, batch_size):
        """
        get batch by batch_size
        :param batch_size:
        :return:
        """
        return self.get_batch(self.next_batch_index(batch_size))

    def update_score(self, index, score):
        """
        dummy methods; for updating scores
        :param index:
        :param score:
        :return:
        """
        pass


class MapPlayback(Playback):

    def __init__(self, capacity, sample_shapes, push_policy="sequence", pop_policy="random",
                 augment_offset={}, augment_scale={}, dtype=None):
        """
        stores map of ndarray.
        returns field '_index' as index of batch samples in sample_batch()
        :param capacity:
        :param sample_shapes:
        :param push_policy:
        :param pop_policy:
        :param dtype:
        """
        super(MapPlayback, self).__init__(capacity, [1], push_policy, pop_policy, dtype)
        self.data = dict([(i, Playback(capacity, sample_shapes[i], push_policy, pop_policy,
                                       augment_offset=augment_offset.get(i), augment_scale=augment_scale.get(i),
                                       dtype=dtype)) for i in sample_shapes])

    def push_sample(self, sample, sample_score=0):
        for i in sample:
            self.data[i].push_sample(sample[i], sample_score)

    def add_sample(self, sample, index, sample_score=0):
        for i in sample:
            self.data[i].add_sample(sample[i], index, sample_score)

    def get_count(self):
        for i in self.data:
            return self.data[i].get_count()

    def get_capacity(self):
        for i in self.data:
            return self.data[i].get_capacity()

    def reset(self):
        for i in self.data:
            self.data[i].reset()

    def next_batch_index(self, batch_size):
        for i in self.data:
            return self.data[i].next_batch_index(batch_size)

    def get_batch(self, index):
        batch = dict([(i, self.data[i].get_batch(index)) for i in self.data])
        batch["_index"] = index
        return batch

    @staticmethod
    def to_rowwise(batch):
        """
        convert column-wise batch to row-wise
        column wise:
            {
                'field_a': [a0, a1, ...],
                'field_b': [b0, b1, ...],
            }
        row wise:[{'field_a': a0, 'field_b': b0}, {'field_a': a1, 'field_b': b1}, ...]
        :param batch:
        :return:
        """
        batch_size = 0
        for i in batch:
            batch_size = len(batch[i])
            break

        row_batch = [{} for _ in range(batch_size)]
        for field in batch:
            data = batch[field]
            for i in range(len(data)):
                row_batch[i][field] = data[i]
        return row_batch

    @staticmethod
    def to_columnwise(batch):
        """
        convert  row-wise batch to column-wise
        row wise:[{'field_a': a0, 'field_b': b0}, {'field_a': a1, 'field_b': b1}, ...]
        column wise:
            {
                'field_a': [a0, a1, ...],
                'field_b': [b0, b1, ...],
            }
        :param batch:
        :return:
        """
        column_batch = {}
        batch_size = len(batch)
        if batch_size == 0:
            return column_batch
        for field in batch[0]:
            column_batch[field] = []
        for i in range(batch_size):
            sample = batch[i]
            for field in sample:
                column_batch[field].append(sample[i])
        for field in column_batch:
            column_batch[field] = np.asarray(column_batch)
        return column_batch


class NearPrioritizedPlayback(MapPlayback):
    """
    using field '_score', typically training error, to store sample score when pushing samples;
    using field '_weight' as priority probability when sample batch from this playback;
    using field '_index' as sample index when sample batch from this playback, for later update_score()
    """
    def __init__(self, capacity, sample_shapes, evict_policy="sequence", epsilon=1e-3,
                 priority_bias=1.0, importance_weight=1.0, dtype=None):
        """

        :param capacity:
        :param sample_shapes:
        :param evict_policy: how old sample is replaced if replay buffer reaches capacity limit.
            "sequence": old sample is replaced as FIFO style;
            "random": old sample is replaced with probability be inversely proportional to sample's 'score_'.
        :param epsilon: minimum score_ regularizer preventing score_ == 0
        :param priority_bias: `alpha`, [0, 1]:  bias introduced from priority sampling.
            can be a constant or a callable for variable value.
            0 for uniform distribution, no bias from priority;
            1 for fully-prioritized distribution, with bias
        :param importance_weight: `beta`, [0, 1]: importance sampling weight correcting priority biases.
            can be a constant or a callable for variable value.
            0 for no correction at all.
            1 for fully compensation for priority bias.
        :param dtype:
        """
        sample_shapes["_score"] = []
        super(NearPrioritizedPlayback, self).__init__(capacity, sample_shapes, "sequence", "random", dtype=dtype)
        self.evict_policy = evict_policy
        self.epsilon, self.priority_bias, self.importance_weight = epsilon, priority_bias, importance_weight

    def push_sample(self, sample, sample_score=None):
        if sample_score is None:
            if self.data["_score"].data is not None:
                sample_score = np.max(self.data["_score"].data)
            else:
                sample_score = 0.0
        print "pushed sample score:", sample_score
        sample["_score"] = np.asarray([float(sample_score)], dtype=float)
        if self.evict_policy == "sequence":
            super(NearPrioritizedPlayback, self).push_sample(sample, sample_score)
        else:
            if self.get_count() < self.get_capacity():
                MapPlayback.push_sample(self, sample)
            else:
                # evict according to score; lower score evict first
                score = self.data["_score"].data
                score = 1 / (score + self.epsilon)
                p = self.compute_distribution(score.reshape(-1))
                index = np.random.choice(np.arange(len(p)), replace=False, p=p)
                # logging.warning("evict sample index:%s, score:%s", index, self.data["_score"].data[index])
                self.add_sample(sample, index)

    def compute_distribution(self, score):
        s_min = np.min(score)
        if s_min < 0:
            score = score - s_min
        exponent = self.priority_bias() if callable(self.priority_bias) else self.priority_bias
        score = np.power(score + self.epsilon, exponent)
        p = score / np.sum(score)
        return p

    def reset(self):
        for i in self.data:
            self.data[i].reset()

    def next_batch_index(self, batch_size):
        if self.get_count() < self.get_capacity():
            p = self.data["_score"].data[:self.get_count()]
        else:
            p = self.data["_score"].data
        p = self.compute_distribution(p.reshape(-1))
        index = np.random.choice(np.arange(len(p)), size=batch_size, replace=False, p=p)
        return index

    def sample_batch(self, batch_size):
        if self.get_count() < self.get_capacity():
            p = self.data["_score"].data[:self.get_count()]
        else:
            p = self.data["_score"].data
        p = self.compute_distribution(p.reshape(-1))
        index = np.random.choice(np.arange(len(p)), size=batch_size, replace=False, p=p)
        priority = p[index]
        batch = super(NearPrioritizedPlayback, self).get_batch(index)
        sample_count = self.get_count()
        is_exponent = self.importance_weight() if callable(self.importance_weight) \
            else self.importance_weight
        w = np.power(sample_count * priority, -is_exponent)
        # global max instead of batch max.
        # todo mathematically, global max is the correct one to use.
        # but in larger replay buffer, this could cause much slower effective learning rate.
        max_all_w = np.power(np.min(p) * sample_count, - is_exponent)
        w = w / max_all_w
        # max_w = np.max(w)
        # if max_w > 1.0:
        #     w = w / np.max(w)

        batch["_index"], batch["_weight"] = index, w
        return batch

    def update_score(self, index, score):
        # logging.warning("update score[%s]: %s -> %s", index, self.data["_score"].data[index], score)
        self.data["_score"].data[index] = score


class NPPlayback(MapPlayback):
    def __init__(self, capacity, sample_shapes, pn_ratio=1.0, push_policy="sequence", pop_policy="random", dtype=np.float32):
        """
        divide MapPlayback into positive sample and negative sample.

        :param capacity:
        :param sample_shapes:
        :param pn_ratio:
        :param push_policy:
        :param pop_policy:
        :param dtype:
        """
        MapPlayback.__init__(self, 1, sample_shapes, push_policy, pop_policy, dtype)
        self.minus_playback = MapPlayback(int(capacity / (1 + pn_ratio)), sample_shapes, push_policy, pop_policy, dtype)
        self.plus_playback = MapPlayback(int(capacity * pn_ratio / (1 + pn_ratio)), sample_shapes, push_policy, pop_policy, dtype)
        self.data = None

    def get_count(self):
        return self.minus_playback.get_count() + self.plus_playback.get_count()

    def get_batch(self, index):
        threshold = self.minus_playback.get_capacity()
        minus_batch = self.minus_playback.get_batch(index[index < threshold])
        plus_batch = self.plus_playback.get_batch(index[index >= threshold] - threshold)
        batch = dict([(i, np.concatenate([minus_batch[i], plus_batch[i]])) for i in plus_batch])
        return batch

    def next_batch_index(self, batch_size):
        if self.plus_playback.get_count() > self.minus_playback.get_count():
            minus_index = self.minus_playback.next_batch_index(batch_size / 2)
            plus_index = self.plus_playback.next_batch_index(batch_size - len(minus_index))
        else:
            plus_index = self.plus_playback.next_batch_index(batch_size / 2)
            minus_index = self.minus_playback.next_batch_index(batch_size - len(plus_index))

        plus_index += self.minus_playback.get_capacity()
        logging.warning("index:-%s, +%s", minus_index, plus_index)
        return np.concatenate((minus_index, plus_index))

    def push_sample(self, sample, sample_score=0):
        if sample_score <= 0:
            self.minus_playback.push_sample(sample, sample_score)
        else:
            self.plus_playback.push_sample(sample, sample_score)

    def reset(self):
        self.minus_playback.reset()
        self.plus_playback.reset()



