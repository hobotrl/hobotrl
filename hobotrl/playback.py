#
# -*- coding: utf-8 -*-

import logging
import numpy as np
# import lsh


class Playback(object):
    def __init__(self, capacity, sample_shape, push_policy="sequence", pop_policy="random", dtype=None):
        """
        stores ndarray.
        :param capacity: total count of samples stored
        :param sample_length: length of a single sample
        :param push_policy: sequence
        :param pop_policy: sequence/random
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
        return self.data[index]

    def pop_batch(self, batch_size):
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

    def __init__(self, capacity, sample_shapes, push_policy="sequence", pop_policy="random", dtype=None):
        """
        stores map of ndarray.
        :param capacity:
        :param sample_shapes:
        :param push_policy:
        :param pop_policy:
        :param dtype:
        """
        super(MapPlayback, self).__init__(capacity, [1], push_policy, pop_policy, dtype)
        self.data = dict([(i, Playback(capacity, sample_shapes[i], push_policy, pop_policy, dtype)) for i in sample_shapes])

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
        return batch


class NearPrioritizedPlayback(MapPlayback):

    def __init__(self, capacity, sample_shapes, epsilon=1e-5, dtype=None):
        sample_shapes["_score"] = [1]
        super(NearPrioritizedPlayback, self).__init__(capacity, sample_shapes, "sequence", "random", dtype)
        self.epsilon = epsilon

    def push_sample(self, sample, sample_score=None):
        if sample_score is None:
            if self.data["_score"].data is not None:
                sample_score = np.max(self.data["_score"].data)
            else:
                sample_score = 0.0
        # print "pushed sample score:", sample_score
        sample["_score"] = np.asarray([float(sample_score)], dtype=float)
        if self.get_count() < self.get_capacity():
            MapPlayback.push_sample(self, sample)
        else:
            # evict according to score; lower score evict first
            p = self.data["_score"].data
            p = 1 / self.norm(p.reshape(-1))
            p = p / np.sum(p)
            index = np.random.choice(np.arange(len(p)), p=p)
            # logging.warning("evict sample index:%s, score:%s", index, self.data["_score"].data[index])
            self.add_sample(sample, index)

    def norm(self, score):
        return score - np.min(score) + self.epsilon

    def reset(self):
        for i in self.data:
            self.data[i].reset()

    def next_batch_index(self, batch_size):
        if self.get_count() < self.get_capacity():
            p = self.data["_score"].data[:self.get_count()]
        else:
            p = self.data["_score"].data
        p = self.norm(p.reshape(-1))
        p = p / np.sum(p)
        index = np.random.choice(np.arange(len(p)), size=batch_size, replace=False, p=p)
        scores = self.data["_score"].data[index]
        # logging.warning("batch score: %s - %s", np.min(scores), np.max(scores))
        return index

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


# class LSHPlayback(Playback):
#     """
#     * expandable on tree nodes
#     * support action exploration decision based on state node
#     """
#     def __init__(self, capacity, sample_length, key_length, key_maker=None, dtype=np.float32):
#         """
#
#         :param capacity:
#         :param sample_length:
#         :param key_length:
#         :param key_maker: function mapping value to key
#         :param dtype:
#         """
#         super(LSHPlayback, self).__init__(1, sample_length, "sequence", "random", dtype)
#         self.space_table = lsh.LSHTree(key_length, 2)
#         if key_maker is None:
#             key_maker = lambda v: v[:key_length]
#         self.key_maker = key_maker
#         self.capacity = capacity
#
#     def push_sample(self, sample, sample_score=0):
#         key = self.key_maker(sample)
#         node = self.space_table.get_bucket(key)
#         if self.space_table.get_count() < self.capacity:
#             node.append_value_array(key, sample, {})
#         else:
#             # table full; need to rebalance
#             print("table full; need to rebalance")
#             node_max = self.space_table.select_node(None, lsh.select_func_dist)
#             if node_max == node:
#                 print("overwriting in node_max")
#                 node.append_value_array(key, sample, {}, overwrite=True)
#             else:
#                 print "shrinking node_max,"
#                 node_max.print_self(4)
#                 node_max.shrink(1)
#                 print "after shrinking node_max,"
#                 node_max.print_self(4)
#                 node.append_value_array(key, sample, {}, overwrite=False)
#         self.space_table.print_self()
#
#     def pop_batch(self, batch_size):
#         return self.space_table.pop(batch_size, lsh.distribution_count)
#
#     def get_count(self):
#         return self.space_table.get_count()
#
#


