# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import unittest
import tempfile
import logging

import numpy as np

from hobotrl.playback import *


class TestCached(unittest.TestCase):
    def test_CachedMap(self):
        cache_path = tempfile.mkdtemp()
        logging.warning("cache path:%s", cache_path)
        playback = CachedMapPlayback(cache_path, 10)
        for i in range(20):
            playback.push_sample({"a": i * np.ones([2], dtype=np.uint8)})
        playback.save()

        playback = CachedMapPlayback(cache_path, 10)
        playback.load()
        self.assertEqual(sum(playback.data["a"].data[1]), sum(np.array([11, 11], dtype=np.uint8)))

    def test_BigMap(self):
        cache_path = tempfile.mkdtemp()
        logging.warning("cache path:%s", cache_path)
        playback = BigMapPlayback(cache_path=cache_path, capacity=16, bucket_size=4)
        for i in range(32):
            playback.push_sample({"a": i * np.ones([2], dtype=np.uint8)})
        playback.save()

        playback = BigMapPlayback(cache_path=cache_path, capacity=16, bucket_size=4, epoch_count=2)
        for i in range(32):
            playback.sample_batch(2)

    def runTest(self):
        self.test_CachedMap()
        self.test_BigMap()

if __name__ == '__main__':
    TestCached().test_BigMap()



