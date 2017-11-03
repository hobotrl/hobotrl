#
# -*- coding: utf-8 -*-

import os
import json
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
        logging.warning("capacity:%s", capacity)
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

            logging.warning("initializing data with: %s, type: %s", sample, type(sample))
            sample_class = type(sample)
            if sample_class in scalar_type:
                sample_shape = []  # scalar value
                if sample_class in dtype_identitical:
                    sample_type = sample_class
                else:
                    sample_type = dtype_mapping[sample_class]
            else:  # try cast as ndarray
                try:
                    sample = np.asarray(sample)
                    sample_shape = list(sample.shape)
                    sample_type = sample.dtype
                except:  # unknown type:
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
        self.add_sample(sample, self.push_index, sample_score)
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

    def get_count(self):
        return self.count

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
        column_shape = {}
        batch_size = len(batch)
        if batch_size == 0:
            return column_batch
        for field in batch[0]:
            column_batch[field] = []
        for i in range(batch_size):
            sample = batch[i]
            for field in sample:
                column_batch[field].append(sample[field])
                if field not in column_shape:
                    column_shape[field] = sample[field].shape
                else:
                    if column_shape[field] != sample[field].shape:
                        logging.error("column[%s] shape mismatch: old:%s, new:%s",
                                      field, column_shape[field], sample[field].shape)
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


class BatchIterator(object):
    def __init__(self, batch, mini_size=1):
        """
        Iterate over a batch of data.
        :param batch: column-wise batch sample.
        :param mini_size: mini batch size, >=1
        """
        super(BatchIterator, self).__init__()
        self._data, self._batch_size = MapPlayback.to_rowwise(batch), mini_size
        self._size = len(self._data)
        self._batch_size = min(self._batch_size, self._size)
        self._next_index = 0

    def __iter__(self):
        return self

    def next(self):
        if self._next_index >= self._size:
            raise StopIteration

        next_index = min(self._next_index, self._size - self._batch_size)
        self._next_index += self._batch_size
        return MapPlayback.to_columnwise(self._data[next_index:next_index+self._batch_size])


class BalancedMapPlayback(MapPlayback):
    """MapPlayback with rebalanced action and done distribution.
    The current balancing method only support discrete action spaces.
    """
    def __init__(self, num_actions, discount=0.995, upsample_bias=None,
                 p_min=None, *args, **kwargs):
        super(BalancedMapPlayback, self).__init__(*args, **kwargs)
        self.NUM_ACTIONS = num_actions
        self.DISCOUNT = discount
        self.P_MIN = p_min if p_min is not None else (1e-3, 1e-2)
        self.UPSAMPLE_BIAS = upsample_bias if upsample_bias is not None \
            else tuple([1.0] * (self.NUM_ACTIONS + 1))

        self.sample_prob =  num_actions * np.ones(self.capacity)
        self.action_prob = 1.0/num_actions*np.ones(num_actions)
        self.done_prob = 0.0

    def next_batch_index(self, batch_size):
        count = self.get_count()
        if count == 0:
            return super(BalancedMapPlayback, self).next_batch_index(batch_size)
        else:
            p = self.sample_prob[:count] / np.sum(self.sample_prob[:count])
            return np.random.choice(
                np.arange(count), size=batch_size, replace=True, p=p
            )

    def push_sample(self, sample, **kwargs):
        index = self.push_index

        # Calculate un-normalized re-sampling weight for sample
        assert 'action' in sample and 'episode_done' in sample
        action = sample['action']
        done = sample['episode_done']
        self.sample_prob[index] = 1 / self.action_prob[action] * \
                                  self.UPSAMPLE_BIAS[action]
        if done:
            self.sample_prob[index] *= 1 / self.done_prob * \
                                       self.UPSAMPLE_BIAS[-1]
        else:
            self.sample_prob[index] *= 1 / (1 - self.done_prob)

        # Exponential moving averaged action and done probability
        delta = np.zeros(self.NUM_ACTIONS)
        delta[action] = 1
        self.action_prob = self.action_prob * self.DISCOUNT + \
                           delta*(1 - self.DISCOUNT)
        cap = self.P_MIN[0]
        self.action_prob[self.action_prob<cap] = cap
        self.action_prob /= np.sum(self.action_prob)

        self.done_prob = self.done_prob*self.DISCOUNT + \
                         float(done)*(1 - self.DISCOUNT)
        cap = self.P_MIN[1]
        self.done_prob = max(min(self.done_prob, 1-cap), cap)

        print ("[BalancedMapPlayback.push_sample()]: "
               "action {}, done {:.3f}, sample {:.3f}").format(
                   self.action_prob, self.done_prob, self.sample_prob[index])

        super(BalancedMapPlayback, self).push_sample(sample, **kwargs)


class CachedMapPlayback(MapPlayback):

    META_FILE = "meta.json"
    NP_POSTFIX = ".npy"

    def __init__(self, cache_path, *args, **kwargs):
        super(CachedMapPlayback, self).__init__(*args, **kwargs)
        # cache_path = kwargs["cache_path"]
        self._path = cache_path
        self._io_status = None  # None, init, loading, flushing, ready
        self.init_()

    def init_(self):
        while True:
            if not os.path.isdir(self._path):
                break
            meta_path = os.sep.join([self._path, self.META_FILE])
            if not os.path.isfile(meta_path):
                break
            data = json.loads(open(meta_path).read())
            self.count = int(data["count"])
            self.push_index = int(data["push_index"])
            break
        self._io_status = "init"

    def load(self):
        self._io_status = "loading"
        while True:
            if not os.path.isdir(self._path):
                break
            paths = os.listdir(self._path)
            paths = filter(lambda p: p.endswith(self.NP_POSTFIX), paths)
            fields = [p[:-len(self.NP_POSTFIX)] for p in paths]
            if len(fields) == 0:
                break
            self.init_data_(fields)
            for field, path in zip(fields, paths):
                self.data[field].data = np.load(os.sep.join([self._path, path]))
            break
        self._io_status = "ready"

    def save(self):
        self._io_status = "flushing"
        if not os.path.isdir(self._path):
            os.makedirs(self._path)
        meta_path = os.sep.join([self._path, self.META_FILE])
        with open(meta_path, mode="w") as f:
            f.write(json.dumps({
                "count": self.count,
                "push_index": self.push_index
            }))
        if self.data is not None:
            for field in self.data:
                data = self.data[field].data
                np.save(os.sep.join([self._path, field+self.NP_POSTFIX]), data)
        self._io_status = "ready"

    def release_mem(self):
        self._io_status = "init"
        self.init_()
        self.data = None
        pass


class BigMapPlayback(Playback):

    META_FILE = "meta.json"

    def __init__(self, *args, **kwargs):
        """Stores maps of ndarray.
        :param capacity:
        :param push_policy:
        :param pop_policy:
        :param dtype:
        :param bucket_size: size of a single bucket.
            There will be at most 2 bucket of data swapped in memory simultaneously.
        :param cache_path:
            path under which to store swapped out data.
        """
        sup_kwargs = self.pick_args(["capacity", "push_policy", "pop_policy", "augment_offset", "augment_scale"], args, kwargs)
        super(BigMapPlayback, self).__init__(**sup_kwargs)
        if "bucket_size" in kwargs:
            self._bucket_size = kwargs["bucket_size"]
            self._bucket_count = self.capacity / self._bucket_size
            del kwargs["bucket_size"]
        else:
            self._bucket_count = 8
            self._bucket_size = self.capacity / self._bucket_count
        if "cache_path" in kwargs:
            self._cache_path = kwargs["cache_path"]
            del kwargs["cache_path"]
        else:
            self._cache_path = "."
        if "epoch_count" in kwargs:
            self._pop_epoch_count = kwargs["epoch_count"]
            del kwargs["epoch_count"]
        else:
            self._pop_epoch_count = 8

        if len(args) > 0:
            args = list(args)
            args[0] = self._bucket_size  # capacity
        elif "capacity" in kwargs:
            kwargs["capacity"] = self._bucket_size
        self._buckets = [CachedMapPlayback(os.sep.join([self._cache_path, str(i)]), *args, **kwargs) for i in range(self._bucket_count)]
        # self._buckets = []
        # for i in range(self._bucket_count):
        #     kwargs.update({"cache_path": os.sep.join([self._cache_path, str(i)])})
        #     self._buckets.append(CachedMapPlayback(*args, **kwargs))

        self._push_bucket = self._pop_bucket = 0
        self._pop_count = 0
        self._last_push_index = 0
        self.init_from_cache_()

    def pick_args(self, fields, args, kwargs):
        picked_args = {}
        for i in range(len(args)):
            picked_args[fields[i]] = args[i]
        for f in fields:
            if f in kwargs:
                picked_args[f] = kwargs[f]
        return picked_args

    def init_from_cache_(self):
        if not os.path.isdir(self._cache_path):
            return
        meta_path = os.sep.join([self._cache_path, self.META_FILE])
        if not os.path.isfile(meta_path):
            return
        data = json.loads(open(meta_path).read())
        self._push_bucket = int(data["push_bucket"])
        self._pop_bucket = int(data["pop_bucket"])
        self._buckets[self._push_bucket].load()
        if self._pop_bucket != self._push_bucket:
            self._buckets[self._pop_bucket].load()

    def save(self):
        if not os.path.isdir(self._cache_path):
            os.makedirs(self._cache_path)
        meta_path = os.sep.join([self._cache_path, self.META_FILE])
        with open(meta_path, mode="w") as f:
            f.write(json.dumps({
                "push_bucket": self._push_bucket,
                "pop_bucket": self._pop_bucket
            }))

    def push_sample(self, sample, sample_score=0):
        self.check_push_bucket_()
        self._buckets[self._push_bucket].push_sample(sample, sample_score)

    def add_sample(self, sample, index, sample_score=0):
        self.check_push_bucket_()
        index = index - (self._bucket_size * self._push_bucket)
        self._buckets[self._push_bucket].add_sample(sample, index, sample_score)

    def get_batch(self, index):
        index = index - (self._bucket_size * self._pop_bucket)
        batch = self._buckets[self._pop_bucket].get_batch(index)
        self.check_pop_bucket_(len(index))
        return batch

    def next_batch_index(self, batch_size):
        index = self._buckets[self._pop_bucket].next_batch_index(batch_size)
        index = index + (self._bucket_size * self._pop_bucket)
        return index

    def reset(self):
        for bucket in self._buckets:
            bucket.release_mem()
        self._push_bucket = self._pop_bucket = 0

    def get_count(self):
        return sum([b.get_count() for b in self._buckets])

    def get_capacity(self):
        return super(BigMapPlayback, self).get_capacity()

    def check_push_bucket_(self):
        current_push_index = self._buckets[self._push_bucket].push_index
        if current_push_index < self._last_push_index:
            # probably push_index rewinded from end of buffer to 0
            self._buckets[self._push_bucket].save()
            new_push = (self._push_bucket + 1) % self._bucket_count
            self._buckets[new_push].load()
            old_push = self._push_bucket
            self._push_bucket = new_push
            self._buckets[old_push].release_mem()
            self._last_push_index = 0
            logging.warning("push index: %s => %s", old_push, new_push)
            self.save()
        else:
            self._last_push_index = current_push_index

    def check_pop_bucket_(self, batch_size):
        self._pop_count += batch_size
        if self._pop_count / self._bucket_size > self._pop_epoch_count:
            # re-select pop bucket
            candidates = np.where([b.get_count() > batch_size for b in self._buckets])[0]
            new_pop = np.random.choice(candidates)
            logging.warning("pop index: %s => %s", self._pop_bucket, new_pop)
            if new_pop != self._pop_bucket:
                # load new_pop
                self._buckets[new_pop].load()
                # release old pop
                if self._pop_bucket != self._push_bucket:  # cannot release while pushing!
                    self._buckets[self._pop_bucket].release_mem()
                self._pop_bucket = new_pop
            self._pop_count = 0



