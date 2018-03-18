# -*- coding: utf-8 -*-

import os
import time
import logging
import traceback
import json
import operator
from collections import deque
from threading import Thread, Event
import wrapt
import numpy as np
from externals.joblib import joblib


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


def shape_of(value):
    if isinstance(value, np.ndarray):
        return value.shape
    return ()


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
            column_batch[field].append(np.asarray(sample[field]))
            if field not in column_shape:
                column_shape[field] = shape_of(sample[field])
            else:
                if column_shape[field] != shape_of(sample[field]):
                    logging.error("column[%s] shape mismatch: old:%s, new:%s",
                                  field, column_shape[field], shape_of(sample[field]))
    for field in column_batch:
        try:
            column_batch[field] = np.asarray(column_batch[field])
        except Exception, e:
            logging.warning("[%s]:type:%s, value:%s", field, type(column_batch[field]), column_batch[field])
            raise e
    return column_batch


class Trajectory(object):
    def __init__(self, max_length=1000):
        super(Trajectory, self).__init__()
        self._max_length = max_length
        self._steps = []
        self._finalized = False

    def transition(self, state, action, reward, next_state, episode_done, **kwargs):
        if not self._finalized:
            if state is not None and \
                action is not None and \
                reward is not None and \
                next_state is not None and \
                episode_done is not None:
                kwargs.update({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "episode_done": episode_done
                })
                self._steps.append(kwargs)
                self._finalized = episode_done or len(self._steps) >= self._max_length
        else:
            raise ValueError("cannot append further transitions!length:%d, finalized:%s"
                             % (len(self._steps), self._finalized))

    @property
    def finalized(self):
        return self._finalized

    def __len__(self):
        return len(self._steps)

    def sample_trajectory(self, start=0, length=None):
        if length is None:
            # sample as long as possible
            end = len(self._steps)
        else:
            # honor length over start
            end = min(start + length, len(self._steps))
            start = end - length
        trajectory = self._steps[start:end]
        trajectory = to_columnwise(trajectory)
        trajectory["_index"] = np.array(range(start, end))
        return trajectory


def strip_args(arg_slots, kwarg_slots, args, kwargs):
    """Strip arguments according to slot specification.
    Strip arguments from `args` and `kwargs` accoding to slot specifications
    and build a key-word argument dictionary.

    Arguments are stripped from `args` according the slot order defined in
    `arg_slots + kwarg_slots`. Then additional argument are stripped from
    `kwargs` according to the slot names.

    :param arg_slots: positional argument slots.
    :param kwarg_slots: key-word argument slots
    :param args: positional arguments.
    :param kwargs:  key-word arguments.
    :return:
    """
    ret_kwargs = {}
    slots = arg_slots + kwarg_slots
    for i, arg in enumerate(args):
        ret_kwargs[slots[i]] = arg
    for key, arg in kwargs.iteritems():
        if key in slots:
            ret_kwargs[key] = arg

    return ret_kwargs


_SCALA_TYPE = [
    bool, int, float,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float, np.float16, np.float32, np.float64]

_DTYPE_IDENTICAL = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float, np.float16, np.float32, np.float64]

_DTYPE_MAPPING = {
    bool: np.bool,
    int: np.int32,
    float: np.float32}

_DTYPE_UNCHANGED = [
    Trajectory
]


class Playback(object):
    _ARG_SLOTS = ('capacity',)
    _KWARG_SLOTS = ('push_policy', 'pop_policy', 'augment_offset',
                    'augment_scale')
    def __init__(self, capacity, push_policy="sequence", pop_policy="random",
                 augment_offset=None, augment_scale=None, *args, **kwargs):
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
        # logging.warning("[Playback.__init__()]: capacity: %s", capacity)
        self.data = None
        self.sample_type = None
        self.sample_shape = None
        self.push_policy = push_policy
        self.pop_policy = pop_policy

        self.augment_offset = 0 if augment_offset is None else augment_offset
        self.augment_scale = 1 if augment_scale is None else augment_scale
        self.count = 0
        self.push_index = 0
        self.pop_index = 0

    def get_count(self):
        """Current count of samples stored

        :return:
        """
        return self.count

    def get_capacity(self):
        """Maximum count of samples can be stored
        :return:
        """

        return self.capacity

    def reset(self):
        """Clear all samples.

        :return:
        """
        self.count, self.push_index, self.pop_index = 0, 0, 0
        self.data = None

    def add_sample(self, sample, index, sample_score=0):
        """Add sample to specified position and modify self.count.

        :param sample:
        :type sample: np.ndarray
        :param index:
        :param sample_score:
        :return:
        """
        # Lazy creation
        if self.data is None:
            # Type and shape check
            sample_class = type(sample)
            if sample_class in _SCALA_TYPE:  # scalar value
                sample_shape = ()
                if sample_class in _DTYPE_IDENTICAL:
                    sample_type = sample_class
                else:
                    sample_type = _DTYPE_MAPPING[sample_class]
            elif sample_class in _DTYPE_UNCHANGED:
                sample_type, sample_shape = None, None
            else:  # try cast as ndarray
                try:
                    cast_sample = np.asarray(sample)
                    sample_shape = tuple(cast_sample.shape)
                    sample_type = cast_sample.dtype
                except:  # unknown type:
                    raise NotImplementedError(
                        "Unsupported sample type:" + str(sample)
                    )
            if self.sample_type is None:
                self.sample_type = sample_type
                self.sample_shape = sample_shape
            logging.warning(
                ("[Playback.add_sample()]: initializing data with: "
                 "class:%s, type: %s, shape: %s"), sample_class, sample_type, sample_shape
            )
            self.data = [None] * self.capacity

        # Place data
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
        if self.sample_shape is not None and self.sample_type is not None:
            ret = np.zeros(shape=[len(index)] + list(self.sample_shape),
                           dtype=self.sample_type)
            for i1, i2 in enumerate(index):
                ret[i1] = np.array(self.data[i2], copy=False)
            return (ret + self.augment_offset) * self.augment_scale
        else:
            ret = [None] * len(index)
            for i1, i2 in enumerate(index):
                ret[i1] = self.data[i2]
            return ret

    def sample_batch(self, batch_size):
        """Sample a batch of samples from buffer.
        :param batch_size:
        :return:
        """
        return self.get_batch(self.next_batch_index(batch_size))

    def update_score(self, index, score):
        """Dummy methods for updating scores.

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

    def init_data(self, sample):
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
            self.init_data(sample)
        # push sample iteratively into Playbacks
        self.add_sample(sample, self.push_index, sample_score)
        self.push_index = (self.push_index + 1) % self.capacity

    def add_sample(self, sample, index, sample_score=0):
        for key in self.data:
            self.data[key].add_sample(sample[key], index, sample_score)
        if self.count < self.capacity:
            self.count += 1

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
    _ARG_SLOTS = ('capacity',)
    _KWARG_SLOTS = ('augment_offset', 'augment_scale', 'evict_policy',
                    'sample_policy',
                    'epsilon', 'priority_bias', 'importance_weight')
    def __init__(self, capacity, augment_offset={}, augment_scale={},
                 evict_policy="sequence", sample_policy="proportional",
                 epsilon=1e-3, priority_bias=1.0, importance_weight=1.0):
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
        self.sample_policy = sample_policy
        self.epsilon, self.priority_bias, self.importance_weight = epsilon, priority_bias, importance_weight

    def push_sample(self, sample, sample_score=None):
        if sample_score is None:
            if self.data is not None and self.data["_score"].data is not None:
                sample_score = np.max(self.data["_score"].data)
                # logging.warning("maxed score: %s", sample_score)
            else:
                sample_score = 0.0
        # logging.warning("pushed sample, score: %s", sample_score)
        sample["_score"] = float(sample_score)
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
        if self.sample_policy == "proportional":
            s_min = np.min(score)
            if s_min < 0:
                score = score - s_min
            exponent = self.priority_bias
            score = np.power(score + self.epsilon, exponent)
        elif self.sample_policy == "rank":
            rank = len(score) - 1 - score.argsort().argsort()
            score = (1.0/(np.arange(len(score))+1))[rank]
        else:
            raise Exception("Not implemented.")
        p = score / np.sum(score)
        return p

    def reset(self):
        for i in self.data:
            self.data[i].reset()

    def next_batch_index(self, batch_size):
        if self.get_count() == 0:
            return np.asarray([], dtype=int)
        elif self.get_count() < self.get_capacity():
            p = self.data["_score"].data[:self.get_count()]
        else:
            p = self.data["_score"].data
        p = self.compute_distribution(np.asarray(p).reshape(-1))
        index = np.random.choice(np.arange(len(p)), size=batch_size, replace=False, p=p)
        return index

    def sample_batch(self, batch_size):
        if self.get_count() < self.get_capacity():
            p = self.data["_score"].data[:self.get_count()]
        else:
            p = self.data["_score"].data
        p = self.compute_distribution(np.array(p).reshape(-1))
        index = np.random.choice(np.arange(len(p)), size=batch_size, replace=False, p=p)
        priority = p[index]
        batch = super(NearPrioritizedPlayback, self).get_batch(index)
        sample_count = self.get_count()
        is_exponent = self.importance_weight
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
        for i, s in zip(index, score):
            self.data["_score"].data[i] = s


class NPPlayback(MapPlayback):
    _ARG_SLOTS = ('capacity',)
    _KWARG_SLOTS = ('pn_ratio', 'push_policy', 'pop_policy')
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
    def __init__(self, batch, mini_size=1, check_episode_done=False):
        """
        Iterate over a batch of data.
        :param batch: column-wise batch sample.
        :param mini_size: mini batch size, >=1
        """
        super(BatchIterator, self).__init__()
        self._data, self._batch_size = to_rowwise(batch), mini_size
        self._size = len(self._data)
        self._batch_size = min(self._batch_size, self._size)
        self._next_index = 0
        self._check_episode_done = check_episode_done

    def __iter__(self):
        return self

    def next(self):
        if self._next_index >= self._size:
            raise StopIteration

        next_index = min(self._next_index, self._size - self._batch_size)
        if self._check_episode_done:
            batch_size = self._batch_size
            for i in range(batch_size):
                if self._data[next_index + i]["episode_done"]:
                    batch_size = i + 1
                    break
            next_end = next_index + batch_size
            pass
        else:
            batch_size = self._batch_size
            next_end = next_index + batch_size
        self._next_index = next_end
        return to_columnwise(self._data[next_index:next_end])


class BalancedMapPlayback(MapPlayback):
    """MapPlayback with rebalanced action and done distribution.
    The current balancing method only support discrete action spaces.
    """
    _ARG_SLOTS = ('num_actions',) + MapPlayback._ARG_SLOTS
    _KWARG_SLOTS = ('upsample_bias', 'p_min') + MapPlayback._KWARG_SLOTS
    def __init__(self, *args, **kwargs):
        kwargs = strip_args(self._ARG_SLOTS, self._KWARG_SLOTS, args, kwargs)
        sup_kwargs = strip_args(MapPlayback._ARG_SLOTS,
                                MapPlayback._KWARG_SLOTS, [], kwargs)
        super(BalancedMapPlayback, self).__init__(**sup_kwargs)

        self.NUM_ACTIONS = kwargs['num_actions']
        self.UPSAMPLE_BIAS = kwargs['upsample_bias'] if 'upsample_bias' in kwargs \
            else tuple([1.0] * (self.NUM_ACTIONS + 1))
        self.P_MIN = kwargs['p_min'] if 'p_min' in kwargs \
            else (1e-3, 1e-2)

        self.cache = {}

    def next_batch_index(self, batch_size):
        count = self.get_count()
        if count == 0:
            return super(BalancedMapPlayback, self).next_batch_index(batch_size)
        else:
            p = self.sample_prob[:count]
            return np.random.choice(
                np.arange(count), size=batch_size, replace=True, p=p
            )

    def push_sample(self, sample, *args, **kwargs):
        self.cache = {}
        super(BalancedMapPlayback, self).push_sample(sample, **kwargs)

    @property
    def sample_prob(self):
        try:
            ret = self.cache['sample_prob']
        except:
            count = self.get_count()
            p_action = self.action_prob / self.UPSAMPLE_BIAS[:-1]
            data = np.array(self.data['action'].data[:count],
                            dtype=np.uint8)
            p_actions = p_action[data,]
            p_done = np.array((1-self.done_prob, self.done_prob)) / \
                np.array((1.0, self.UPSAMPLE_BIAS[-1]))
            data = np.array(self.data['episode_done'].data[:count],
                            dtype=np.uint8)
            p_dones = p_done[data]
            sample_prob = 1.0 / p_actions / p_dones
            sample_prob /= np.sum(sample_prob)
            if count == self.capacity:
                self.cache['sample_prob'] = sample_prob
            ret = sample_prob
        return ret

    @property
    def done_prob(self):
        try:
            ret = self.cache['done_prob']
        except:
            count = self.get_count()
            data = self.data['episode_done'].data[:count]
            prob = 1.0 * sum(data) / self.count
            cap = self.P_MIN[1]
            prob = np.maximum(np.minimum(prob, 1-cap), cap)
            if count == self.capacity:
                self.cache['done_prob'] = prob
            ret = prob
        return ret

    @property
    def action_prob(self):
        try:
            ret = self.cache['action_prob']
        except:
            count = self.get_count()
            data = np.array(self.data['action'].data[:count])
            one_hot = np.arange(self.NUM_ACTIONS)[:, np.newaxis] == \
                      data[np.newaxis, :]
            cnt = np.sum(one_hot, axis=1)
            prob = 1.0 * cnt / np.sum(cnt)
            cap = self.P_MIN[0]
            prob = np.maximum(np.minimum(prob, 1-cap), cap)
            if count == self.capacity:
                self.cache['action_prob'] = prob
            ret = prob
        return prob


class PersistencyWrapper(wrapt.ObjectProxy):
    """Pass-through Persistency Wrapper for Playbacks.
    This wrapper class keeps a disk copy of the wrapped Playback. Saved data
    can be loaded back to memory and the sampling and push related counters
    are guaranteed to be consistent.

    The `count` and `push_index` attributes are kept as meta data to keep
    consistency of a resumed session. No other attributes (e.g. capacity) are
    checked, which means the disk copy can be converted to any compatible
    in-memory Playback.
    """
    META_FILE = "meta.json"
    DATA_FILE = "data.pkl"
    def __init__(self, playback, path, *args, **kwargs):
        """ Initialization.

        :param playback_cls: the playback instance to be wrapped.
        :param path: path to persist data.
        :param args: dummy
        :param kwargs: dummy
        """
        super(PersistencyWrapper, self).__init__(playback)
        self._path = path
        self._io_status = None  # None, init, loading, flushing, ready
        self.load_meta()

    def load_meta(self):
        while True:
            if not os.path.isdir(self._path):
                break
            meta_path = os.sep.join([self._path, self.META_FILE])
            if not os.path.isfile(meta_path):
                break
            try:
                # logging.warning(
                #     "[PersistencyWrapper.__load_meta()]: "
                #     "loading meta data from path: %s", meta_path
                # )
                data = json.loads(open(meta_path).read())
                self.__wrapped__.count = int(data["count"])
                self.__wrapped__.push_index = int(data["push_index"])
                # logging.warning(
                #     "[PersistencyWrapper.__load_meta()]: "
                #     "loaded meta data: count {}, push_index {}".format(
                #         self.__wrapped__.count,
                #         self.__wrapped__.push_index
                #     )
                # )
                break
            except:
                logging.warning(
                    "[PersistencyWrapper.__load_meta()]: "
                    "exception loading meta data, retry in one second."
                )
                time.sleep(1.0)

        self._io_status = "init"

    def load(self):
        self._io_status = "loading"
        # logging.warning(
        #     "[PersistencyWrapper.load()]: "
        #     "loading data from path %s...", self._path
        # )
        self.load_meta()
        if self.__wrapped__.count > 0:
            try:
                path = os.sep.join([self._path, self.DATA_FILE])
                if os.path.isfile(path):
                    self.__wrapped__.data = joblib.load(path)
                    # logging.warning(
                    #     "[PersistencyWrapper.load()]: "
                    #     "loaded data from {}.".format(path)
                    # )
                else:
                    logging.warning(
                        "[PersistencyWrapper.load()]: "
                        "data file {} does not exist yet.".format(path)
                    )                    
            except Exception, e:
                logging.warning(
                    "[PersistencyWrapper.load()]: "
                    "load data from {} failed.".format(path)
                )
                logging.warning(traceback.format_exc())
                raise Exception("loading data from {} failed.".format(path))
        else:
            pass
        self._io_status = "ready"

    def save(self):
        """Save meta and data.
        Save meta data after data is saved to make this a transaction.
        """
        self._io_status = "flushing"
        logging.info(
            "[PersistencyWrapper.save()] : "
            "saving data to path %s...", self._path
        )
        if not os.path.isdir(self._path):
            os.makedirs(self._path)

        if self.data is not None:
            path = os.sep.join([self._path, self.DATA_FILE])
            joblib.dump(self.__wrapped__.data, path)

        meta_path = os.sep.join([self._path, self.META_FILE])
        with open(meta_path, mode="w") as f:
            f.write(json.dumps({
                "count": self.__wrapped__.count,
                "push_index": self.__wrapped__.push_index
            }))

        self._io_status = "ready"

    def release_mem(self):
        self._io_status = "init"
        self.__wrapped__.reset()
        self.load_meta()
        # logging.warning(
        #     "[PersistencyWrapper.release_mem()]: "
        #     "releasing cache memory...")

    def reset(self):
        self.release_mem()
        self.__wrapped__.reset()


class BigPlayback(Playback):
    """Memory-cached Collection of Persisted Playbacks."""
    META_FILE = "meta.json"
    _ARG_SLOTS = ('bucket_cls',)
    _KWARG_SLOTS = ('bucket_size', 'cache_path', 'max_sample_epoch',
                    'ratio_active') + Playback._ARG_SLOTS + \
                   Playback._KWARG_SLOTS
    def __init__(self, *args, **kwargs):
        """Initialization."""
        # Strip arguments
        #  first strip arguments for myself.
        self_kwargs = strip_args(self._ARG_SLOTS, self._KWARG_SLOTS, args, kwargs)
        #  strip the stripped kwargs again for super class. use empty args to
        #  avoid confusion.
        sup_kwargs = strip_args(
            Playback._ARG_SLOTS, Playback._KWARG_SLOTS, [], self_kwargs
        )

        # Build superclass
        super(BigPlayback, self).__init__(**sup_kwargs)

        # Initialize self attributes
        self._bucket_cls = None  # class of buckets
        self._bucket_size = self._bucket_count = None
        self._cache_path = None
        self._max_sample_epoch = None  # average sampling epoch for a bucket
        self._max_active_buckets = None  # desired # of in memory buckets
        self._buckets = []
        self.__init_attrs(self_kwargs, kwargs)

        # Initialize volatile data structures
        self._push_bucket = 0
        self._buckets_active = {i: False for i in range(len(self._buckets))}
        self._buckets_loading = {i: False for i in range(len(self._buckets))}
        self._buckets_saving = {i: False for i in range(len(self._buckets))}
        self._buckets_sample_quota = {}  # remaining sample quota for active buckets
        self._buckets_to_save = deque()
        self._buckets_to_load = deque()
        self._close_flag = False
        self._monitor_stop_event = Event()
        self._monitor_stop_event.clear()

        # helper counters
        self.last_t_getbatch = time.time()
        self.last_t_maintain = time.time()
        self.last_t_bktio = time.time()
        self.cnt_qi_empty = 0
        self.cnt_qo_empty = 0
        
        self._thread_io_monitor = Thread(
            target=self.__monitor_loop, args=(self._monitor_stop_event,)
        )
        self._thread_io_monitor.start()
        # Read back state from disk
        self.__init_from_cache()

    def push_sample(self, sample, sample_score=0):
        """Push new sample into buffer.
        This should be the only way new samples can be inserted into BigMap.
        """
        # Get the bucket id to push and the relative index inside that bucket
        bucket_to_push, rel_index = divmod(self.push_index, self._bucket_size)
        assert bucket_to_push == self._push_bucket
        swap_flag = self._buckets[bucket_to_push].capacity == (rel_index + 1)

        # do the actual sample insertion
        while self._buckets[bucket_to_push]._io_status != 'ready':
            logging.warning(
                "[BigPlayback.push_sample()]: "
                "Waiting push bucket {} to be ready".format(bucket_to_push)
            )
            time.sleep(1.0)

        # logging.warning(
        #     "[BigPlayback.push_sample()]: "
        #     "push into bucket {} @ {}".format(bucket_to_push, self._buckets[bucket_to_push].push_index)
        # )

        try:
            self._buckets[bucket_to_push].push_sample(sample, sample_score)
        except:
            logging.warning(
                traceback.format_exc()
            )

        # adjust push_bucket
        if swap_flag:
            self._buckets_saving[self._push_bucket] = True
            self._buckets_to_save.appendleft(self._push_bucket)
            nxt_push_bucket = (self._push_bucket + 1) % self._bucket_count
            self._push_bucket = nxt_push_bucket
            logging.info(
                "[BigPlayback.push_sample()]: "
                "forwarding push index to {}".format(self._push_bucket)
            )

            nxt_push_bucket = (self._push_bucket + 1) % self._bucket_count
            self._buckets_loading[nxt_push_bucket] = True
            self._buckets_to_load.appendleft(nxt_push_bucket)

    def add_sample(self, sample, index, sample_score=0):
        raise NotImplementedError(
            'We do not support adding sample by index. Use push_sample().'
        )

    def get_batch(self, index):
        ret = []
        for bkt_id, rel_index in index:
            if len(rel_index) == 0:
                continue
            else:
                # Sample from this bucket
                bkt_ret = self._buckets[bkt_id].get_batch(rel_index)
                ret.append(bkt_ret)

            # Leave current and next push bucket alone
            if bkt_id == (self._push_bucket - 1) % self._bucket_count or \
                bkt_id == self._push_bucket or \
                bkt_id == (self._push_bucket + 1) % self._bucket_count:
                continue
            elif self._buckets_saving[bkt_id]:
                continue
            else:
                # Modify sample quota for this bucket
                self._buckets_sample_quota[bkt_id] -= len(rel_index) if self._buckets_sample_quota[bkt_id] > 0 else 0
                # check activation state
                if self._buckets_sample_quota[bkt_id] <= 0:
                    logging.warning(
                        "[BigPlayback.get_batch()]: "
                        "bucket {} has run out of quota.".format(bkt_id)
                    )
                    # deactivate this bucket
                    self._buckets_active[bkt_id] = False

                    # release mem and signal IO thread to load a new bucket.
                    self._buckets[bkt_id].release_mem()
                    self.__maintain_active_buckets()

        ret = self._merge_batches(ret)

        if time.time() - self.last_t_getbatch > 60:
            logging.info(
                "[BigPlayback.get_batch()]: "
                "get batch function alive. Last len (batches) {}".format(len(ret))
            )
            self.last_t_getbatch = time.time()

        return ret

    def _merge_batches(self, batches):
        """
        supports batches from MapPlayback or Playback.
        :param batches:
        :return:
        """
        if len(batches) == 0:
            return {}
        if isinstance(batches[0], dict):
            # column-wise batch; concat each column
            ret = {}
            for bkt_ret in batches:
                for k, v in bkt_ret.iteritems():
                    ret[k] = [] if k not in ret else ret[k]
                    ret[k].append(v)
            for k in ret:
                ret[k] = np.concatenate(ret[k], axis=0)
            return ret
        elif isinstance(batches[0], list):
            # concat into a list
            return sum(batches, [])
        elif isinstance(batches[0], np.ndarray):
            # concat into a larger batch
            return np.concatenate(batches, axis=0)

    def next_batch_index(self, batch_size):
        ret = self.__next_batch_index(batch_size)
        if sum([len(idx) for _, idx in ret]) == batch_size:
            return ret
        else:
            logging.warning(
                "[BigPlayback.next_batch_index]: "
                "number of samples < batch size."
            )
            return []

    def __monitor_loop(self, stop_event, *args, **kwargs):
        _thread_io = None
        while not stop_event.is_set():
            if _thread_io is None or not _thread_io.is_alive():
                logging.warning(
                    "[BigPlayback.monitor_loop()]: "
                    "io thread not running, respawning..."
                )
                _stop_event = Event()
                _stop_event.clear()
                _thread_io = Thread(
                    target=self.bucket_io, args=(_stop_event,),
                    name='thread_io', group=None,
                )
                _thread_io.start()
            else:
                logging.warning(
                    "[BigPlayback.monitor_loop()]: "
                    "io thread running okay!"
                )
            time.sleep(60.0)
        logging.warning(
            "[BigPlayback.monitor_loop()]: stopping io thread."
        )
        _stop_event.set()
        _thread_io.join()
        logging.warning(
            "[BigPlayback.monitor_loop()]: quiting monitoring thread."
        )
        return

    def __next_batch_index(self, batch_size):
        # get an ordered list of active bucket ids.
        list_bkt_active = [k for k, v in self._buckets_active.iteritems() if v]
        cnt_bkt_active = [self._buckets[bkt_id].count for bkt_id in list_bkt_active]
        sample_per_bkt = np.random.multinomial(
            batch_size,
            1.0 * np.array(cnt_bkt_active) / (sum(cnt_bkt_active) + 1e-5)
        )
        ret = []
        for sub_size, bkt_id in zip(sample_per_bkt, list_bkt_active):
            rel_index = tuple(self._buckets[bkt_id].next_batch_index(sub_size))
            ret.append((bkt_id, rel_index))
        return ret

    def __init_attrs(self, self_kwargs, kwargs):
        self._bucket_cls = self_kwargs['bucket_cls']

        if "bucket_size" in self_kwargs:
            self._bucket_size = self_kwargs["bucket_size"]
            self._bucket_count = int(self.capacity / self._bucket_size)
        else:
            self._bucket_count = 10
            self._bucket_size = int(self.capacity / self._bucket_count)
        logging.warning(
            "[BigPlayback.init()]: "
            "Using {} buckets with size {}".format(
                self._bucket_count, self._bucket_size
            )
        )

        if "cache_path" in self_kwargs:
            self._cache_path = self_kwargs["cache_path"]
        else:
            self._cache_path = "./ReplayMemoryData"
        logging.warning(
            "[BigPlayback.init()]: "
            "Using cache path {}".format(self._cache_path)
        )

        if "max_sample_epoch" in self_kwargs:
            self._max_sample_epoch = self_kwargs["max_sample_epoch"]
        else:
            self._max_sample_epoch = 8
        logging.warning(
            "[BigPlayback.init()]: "
            "Max average sample epoch per sample: {}.".format(
                self._max_sample_epoch)
        )

        if "ratio_active" in self_kwargs:
            ratio_active = self_kwargs["ratio_active"]
            assert 1.0/self._bucket_count <= ratio_active <= 1.0
        else:
            ratio_active = 1.0
        self._max_active_buckets = int(self._bucket_count * ratio_active)
        logging.warning(
            "[BigPlayback.init()]: "
            "Number of active buckets is {}.".format(
                self._max_active_buckets)
        )

        # build buckets
        sub_kwargs = strip_args(self._bucket_cls._ARG_SLOTS,
                                self._bucket_cls._KWARG_SLOTS,
                                [], kwargs)
        sub_kwargs['capacity'] = self._bucket_size
        for i in range(self._bucket_count):
            sub_path = os.sep.join([self._cache_path, 'bucket_{}'.format(i)])
            bucket = PersistencyWrapper(self._bucket_cls(**sub_kwargs), sub_path)
            self._buckets.append(bucket)

    def __init_from_cache(self):
        # Load meta data
        if not os.path.isdir(self._cache_path):
            logging.warning(
                "[BigPlayback.init_from_cache_()]: "
                "cache path {} does not exist.".format(self._cache_path)
            )
        else:
            meta_path = os.sep.join([self._cache_path, self.META_FILE])
            if not os.path.isfile(meta_path):
                logging.warning(
                    "[BigPlayback.init_from_cache_()]: "
                    "meta file {} does not exist.".format(meta_path)
                )
            else:
                data = json.loads(open(meta_path).read())
                capacity = int(data['capacity'])
                assert self.capacity == capacity
                self._push_bucket = int(data["push_bucket"])

        # Load bucket meta
        for bkt in self._buckets:
            bkt.load_meta()

        # Load current and next push bucket
        load_buckets = [self._push_bucket]
        next_push_bucket = (self._push_bucket + 1) % self._bucket_count
        load_buckets.append(next_push_bucket)
        for bkt in load_buckets:
            self._buckets_loading[bkt] = True
            self._buckets_to_load.append(bkt)
        while True:
            if all([bid in self._buckets_active for bid in load_buckets]):
                break
            else:
                logging.warning(
                    "[BigPlayback.init_from_cache_()]: "
                    "loading buckets {}".format(load_buckets)
                )
                time.sleep(1.0)
        logging.warning(
            "[BigPlayback.init_from_cache_()]: "
            "loaded buckets {} to push at {} (global {}).".format(
                self._push_bucket,
                self._buckets[self._push_bucket].push_index,
                self.push_index
            )
        )
        self.__maintain_active_buckets()

    def __maintain_active_buckets(self):
        """Maintain a number of active buckets in memory.
        This private method maintains a number of active buckets in memory.
        It checks the diff between desired and actual amount of active
        buckets, as well as the number of persisted buckets that can be
        activated. Then it signal IO thread to load a number of buckets and
        return without waiting.
        """
        # Is there enough active buckets inside memory? How many is due?
        # Total number of buckets to load is the:
        #  #(desired amount) - #(activated and maintained)
        a_bkts = [k for k, v in self._buckets_active.iteritems() if v]
        m_bkts = [k for k, v in self._buckets_loading.iteritems() if v]
        num_to_load = min(
            self._max_active_buckets, int(self.count / self._bucket_size)
        )
        num_to_load -= len(set(a_bkts + m_bkts))
        if num_to_load <= 0:
            return

        # How many can be loaded from disk?
        available_buckets = self.__check_persisted_buckets()
        num_to_load = min(num_to_load, len(available_buckets))

        # Sample from available buckets without replacement
        if num_to_load > 0:
            load_buckets = np.random.choice(
                available_buckets, num_to_load, replace=False
            )
            load_buckets.sort()
            logging.warning(
                "[BigPlayback.__maintain_active_buckets()]: "
                "maintaining active buckets. Buckets to load "
                "are {}".format(load_buckets)
            )
            # put into load queue and let IO thread to load bucket.
            for bkt in load_buckets:
                self._buckets_loading[bkt] = True
                self._buckets_to_load.append(bkt)

        if time.time() - self.last_t_maintain > 60:
            logging.warning(
                "[BigPlayback.__maintain_active_buckets()]: "
                "maintain function alive. Last Num to load {}".format(num_to_load)
            )
            self.last_t_maintain = time.time()

    def __save_meta(self):
        if not os.path.isdir(self._cache_path):
            logging.warning(
                "[BigPlayback.save()]:"
                "making cache path {} .".format(self._cache_path)
            )
            os.makedirs(self._cache_path)
        meta_path = os.sep.join([self._cache_path, self.META_FILE])
        with open(meta_path, mode="w") as f:
            logging.warning(
                "[BigPlayback.__save_meta()]: "
                "writing to meta file {} .".format(meta_path)
            )
            f.write(json.dumps({
                "capacity": self.capacity,
                "push_bucket": self._push_bucket,
            }))

    def __check_persisted_buckets(self):
        """Return ids of non-empty and non-active buckets."""
        return [
            bid for bid, bkt in enumerate(self._buckets)
            if bkt.count > 0 \
               and not self._buckets_active[bid] \
               and not self._buckets_loading[bid] \
               and not self._buckets_saving[bid]
        ]

    @property
    def count(self):
        """Count total number of samples in buckets.
        This overrides the `count` attribute of super class Playback and is
        calculated on-the-run to avoid meta inconsistency due to restart.
        """
        return sum([bkt.count for bkt in self._buckets])

    @property
    def push_index(self):
        """Calculate the next global index to push samples.
        This overrides the `push_index` attribute of super class Playback and is
        calculated on-the-run to avoid meta inconsistency due to restart.
        """
        return  self._bucket_size * self._push_bucket + \
                self._buckets[self._push_bucket].push_index

    @count.setter
    def count(self, dummy):
        logging.warning(
            "[BigPlayback.count]: this is a dummy setter."
        )
        pass

    @push_index.setter
    def push_index(self, dummy):
        logging.warning(
            "[BigPlayback.push_index]: this is a dummy setter."
        )
        pass

    def bucket_io(self, stop_event):
        """Thread function for bucket IO.
        Monitors two queues: `self._buckets_to_save` and `self._buckets_to_load`.
        Save/load bucket to/from disk from/to memory if there is bucket id data
        in those tow queues.

        When loading buckets to memory, also sets `self._buckets_active` and
        assigns sampling quota to  `self._buckets_sample_quota` to
        enable sampling from this bucket.

        return from this thread if `self._close_flag` is set to True.
        """
        while not stop_event.is_set():
            try:
                # Save buckets
                try:
                    if len(self._buckets_to_save) > 0:
                        bkt = self._buckets_to_save.popleft()
                        logging.warning(
                            "[BigPlayback.bucket_io()]: "
                            "notified to save bucket {}.".format(bkt)
                        )
                        try:
                            self.__save_one(bkt)
                        except:
                            logging.warning(traceback.format_exc())
                            logging.warning(
                                "[BigPlayback.bucket_io()]: "
                                "exception saving bucket {}.".format(bkt)
                            )
                        finally:
                            # deactivate this bucket no matter what
                            # if the transaction is not finished leave as is.
                            self._buckets_saving[bkt] = False
                except IndexError:
                    pass
                    self.cnt_qo_empty += 1

                # Load buckets
                try:
                    if len(self._buckets_to_load) > 0:
                        bkt = self._buckets_to_load.popleft()
                        logging.warning(
                            "[BigPlayback.bucket_io()]: "
                            "notified to load bucket {}.".format(bkt)
                        )
                        try:
                            self.__load_one(bkt)
                        except:
                            logging.warning(traceback.format_exc())
                            logging.warning(
                                "[BigPlayback.bucket_io()]: "
                                "exception loading bucket {}.".format(bkt)
                            )
                        finally:
                            self._buckets_loading[bkt] = False
                except IndexError:
                    pass
                    self.cnt_qi_empty += 1
            except:
                logging.warning(
                    "[BigPlayback.bucket_io()]: "
                    "step exception:"
                )
                logging.warning(
                    traceback.format_exc()
                )
            finally:
                time.sleep(0.1)

                if time.time() - self.last_t_bktio > 60:
                    logging.info(
                        "[BigPlayback.bucket_io()]: "
                        "bucket_io function alive. Qi {} Qo {}".format(
                            self.cnt_qi_empty, self.cnt_qo_empty)
                    )
                    self.last_t_bktio = time.time()
                    self.cnt_qi_empty = 0
                    self.cnt_qo_empty = 0

        logging.warning(
            "[BigPlayback.bucket_io()]: returning from IO thread."
        )

    def __load_one(self, bucket_id):
        logging.warning(
            "[BigPlayback.__load_one()]: "
            "going to load bucket {}.".format(bucket_id)
        )
        self._buckets[bucket_id].load()

        # assign sampling quota to this bucket
        self._buckets_sample_quota[bucket_id] = \
            self._max_sample_epoch * self._buckets[bucket_id].count
        self._buckets_active[bucket_id] = True
        logging.warning(
            "[BigPlayback.__load_one()]: "
            "loaded bucket {} into mem.".format(bucket_id)
        )

    def __save_one(self, bucket_id):
        logging.warning(
            "[BigPlayback.__save_one()]: "
            "going to save bucket {}.".format(bucket_id)
        )
        self._buckets[bucket_id].save()
        # Sync meta to truly persist the saved meta.
        # Otherwise the saved bucket data will be ignored in the next
        #  load.
        # TODO: still this won't fully prevent inconsistency btw. the
        #  meta of BigPlayback and its buckets. Should double
        #  check at initialization to prevent this.
        self.__save_meta()
        logging.warning(
            "[BigPlayback.__save_one()]: "
            "saved meta of bucket {}.".format(bucket_id)
        )

    def close(self):
        """Release memory and close IO thread."""
        self.reset()             # release bucket memory
        self._close_flag = True  # signal IO thread to return
        self._monitor_stop_event.set()
        self._thread_io_monitor.join()

    def reset(self):
        """Release bucket memory and reset super class."""
        for bucket in self._buckets:
            bucket.release_mem()
        self._push_bucket = 0
        super(BigPlayback, self).reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


