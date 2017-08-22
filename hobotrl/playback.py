#
# -*- coding: utf-8 -*-

import logging
import numpy as np

scalar_type = [
    bool, int, float,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float, np.float16, np.float32, np.float64]

dtype_identitical = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float, np.float16, np.float32, np.float64]

dtype_mapping = {
    bool: np.bool,
    int: np.int32,
    float: np.float32}


class Playback(object):
    def __init__(self, capacity, push_policy="sequence", pop_policy="random",
                 augment_offset=None, augment_scale=None):
        """
        Stores ndarray, optionally apply transformation:
            (sample + offset) * scale
        to samples before returning.
        :param capacity: total count of samples stored
        :param push_policy: sequence
        :param pop_policy: sequence/random
        :param augment_offset:
        :param augment_scale:
        """
        self.capacity = capacity
        print "capacity:", capacity
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
        """Clear all samples
        :return:
        """
        self.count, self.push_index, self.pop_index = 0, 0, 0

    def add_sample(self, sample, index, sample_score=0):
        """
        add sample to specified position and modify self.count.
        :param sample:
        :type sample: np.ndarray
        :param index:
        :param sample_score:
        :return:
        """
        if self.data is None:
            # lazy creation
            print "initializing data with:", sample, ",type:", type(sample)
            sample_class = type(sample)
            if sample_class == np.ndarray:
                sample_shape = list(sample.shape)
                sample_type = sample.dtype
            elif sample_class in scalar_type:
                sample_shape = []  # scalar value
                if sample_class in dtype_identitical:
                    sample_type = sample_class
                else:
                    sample_type = dtype_mapping[sample_class]
            else:  # unknown type:
                raise NotImplementedError("unsupported sample type:" + str(sample))

            self.data = np.zeros(
                shape=([self.capacity] + sample_shape), dtype=sample_type)

        self.data[index] = sample
        if self.count < self.capacity:
            self.count += 1

    def push_sample(self, sample, sample_score=0):
        """Put sample into buffer and increment push index by 1.
        :param sample:
        :param sample_score:
        :return:
        """
        self.add_sample(sample, self.push_index, sample_score)
        self.push_index = (self.push_index + 1) % self.capacity

    def next_batch_index(self, batch_size):
        """Generate sample index of the next batch.
        :param batch_size:
        :return:
        """
        if self.get_count() == 0:
            return np.asarray([], dtype=int)
        elif self.pop_policy == "random":
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
        """Get batch by index
        :param index:
        :return:
        """
        return (self.data[index] + self.augment_offset) * self.augment_scale

    def sample_batch(self, batch_size):
        """Sample a batch of samples from buffer.
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

    def __init__(self, *args, **kwargs):
        """Stores map of ndarray.
        Contains a dummy inherited Playback and a dict of Playbacks
        :param capacity:
        :param push_policy:
        :param pop_policy:
        :param dtype:
        """
        super(MapPlayback, self).__init__(*args, **kwargs)
        # overide initialized parameters of parent class
        if self.augment_offset == 0:
            self.augment_offset = {}
        if self.augment_scale == 1:
            self.augment_scale = {}

    def init_data_(self, sample):
        """Initialize a dict of {key: Playback} as data."""
        self.data = dict(
            [(key, Playback(
                self.capacity, self.push_policy, self.pop_policy,
                self.augment_offset[key] if key in self.augment_offset else None,
                self.augment_scale[key] if key in self.augment_scale else None))
             for key in sample])

    def push_sample(self, sample, sample_score=0):
        # init key->Playback map
        if self.data is None:
            self.init_data_(sample)
        # push sample iteritively into Playbacks
        for key in sample:
             self.data[key].push_sample(sample[key], sample_score)
        # increment ego count and push_index
        self.add_sample(None, None, None)
        self.push_index = (self.push_index + 1) % self.capacity

    def add_sample(self, sample, index, sample_score=0):
        if self.count < self.capacity:
            self.count += 1
        if sample is None:
            # If caller is push_sample do nothing further.
            return
        else:
            for key in self.data:
                self.data[key].add_sample(sample[key], index, sample_score)

    def get_batch(self, index):
        batch = dict([(key, self.data[key].get_batch(index)) for key in self.data])
        batch["_index"] = index
        return batch

    def reset(self):
        if self.data is None:
            return
        else:
            for i in self.data:
                self.data[i].reset()
            self.count, self.push_index, self.pop_index = 0, 0, 0

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
                column_batch[field].append(sample[field])
        for field in column_batch:
            column_batch[field] = np.asarray(column_batch[field])
        return column_batch


import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class NearPrioritizedPlayback(MapPlayback):
    """
    using field '_score', typically training error, to store sample score when pushing samples;
    using field '_weight' as priority probability when sample batch from this playback;
    using field '_index' as sample index when sample batch from this playback, for later update_score()
    """
    def __init__(self, capacity, augment_offset={}, augment_scale={},
                 evict_policy="sequence", epsilon=1e-3,
                 priority_bias=1.0, importance_weight=1.0):
        """

        :param capacity:
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
        super(NearPrioritizedPlayback, self).__init__(capacity, "sequence", "random",
                                                      augment_offset=augment_offset,
                                                      augment_scale=augment_scale,
                                                      )
        self.evict_policy = evict_policy
        self.epsilon, self.priority_bias, self.importance_weight = epsilon, priority_bias, importance_weight

    def push_sample(self, sample, sample_score=None):
        if sample_score is None:
            if self.data is not None and self.data["_score"].data is not None:
                sample_score = np.max(self.data["_score"].data)
            else:
                sample_score = 0.0
        print "pushed sample score:", sample_score
        sample["_score"] = np.asarray([float(sample_score)], dtype=float)
        if self.evict_policy == "sequence":
            super(NearPrioritizedPlayback, self).push_sample(sample, sample_score)
        else:
            if self.get_count() < self.get_capacity():
                super(NearPrioritizedPlayback, self).push_sample(sample, sample_score)
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
    def __init__(self, capacity, pn_ratio=1.0, push_policy="sequence", pop_policy="random"):
        """
        divide MapPlayback into positive sample and negative sample.

        :param capacity:
        :param pn_ratio:
        :param push_policy:
        :param pop_policy:
        :param dtype:
        """
        super(NPPlayback, self).__init__(1, push_policy, pop_policy)
        self.minus_playback = MapPlayback(int(capacity / (1 + pn_ratio)), push_policy, pop_policy)
        self.plus_playback = MapPlayback(int(capacity * pn_ratio / (1 + pn_ratio)), push_policy, pop_policy)
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


class BalancedMapPlayback(MapPlayback):
    """MapPlayback with rebalanced action and done distribution.
    The current balancing method only support discrete action spaces.
    """
    def __init__(self, num_actions, *args, **kwargs):
        super(BalancedMapPlayback, self).__init__(*args, **kwargs)
        self.num_actions = num_actions
        self.sample_prob =  num_actions * np.ones(self.capacity)
        self.action_prob = 1.0/num_actions*np.ones(num_actions)
        self.done_prob = 0.0

    def next_batch_index(self, batch_size):
        count = self.get_count()
        if count == 0:
            return super(
                BalancedMapPlayback, self).next_batch_index(batch_size)
        else:
            p = self.sample_prob[:count] / np.sum(self.sample_prob[:count])
            return np.random.choice(
                np.arange(count), size=batch_size, replace=True, p=p)

    def push_sample(self, sample, **kwargs):
        index = self.push_index

        # Calculate unnormalized resampling weight for sample
        assert 'action' in sample and 'episode_done' in sample
        action = sample['action']
        done = sample['episode_done']
        if done:
            self.sample_prob[index] = self.num_actions
            self.sample_prob[index] = 1/self.done_prob
        else:
            self.sample_prob[index] = 1/self.action_prob[action]
            self.sample_prob[index] *= 1/(1-self.done_prob)

        # Exponetial moving averaged action and doneprobability
        delta = np.zeros(self.num_actions)
        delta[action] = 1
        self.action_prob = self.action_prob*0.95 + delta*0.05
        cap = 1e-2
        self.action_prob[self.action_prob<cap] = cap
        self.action_prob /= np.sum(self.action_prob)

        self.done_prob = self.done_prob*0.95 + float(done)*0.05
        cap = 1e-1
        if self.done_prob < cap:
            self.done_prob = cap
        if self.done_prob > 1-cap:
            self.done_prob = 1 - cap
            print ("[BalancedMapPlayback.push_sample()]: "
                   "action {}, done {:.3f}, sample {:.3f}").format(
                       self.action_prob, self.done_prob, self.sample_prob[index])

        super(BalancedMapPlayback, self).push_sample(sample, **kwargs)


