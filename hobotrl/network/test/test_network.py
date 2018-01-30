# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import unittest
import tempfile
import logging

import numpy as np
from hobotrl.network import *


class TestNetwork(unittest.TestCase):
    def test_share_weight(self):
        tf.reset_default_graph()
        def f(inputs):
            x = inputs[0]
            y = Utils.layer_fcs(x, [4], 1, var_scope="f")
            y = tf.squeeze(y, axis=-1)
            return {"y": y}
        with tf.name_scope("inputs"):
            x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        net1 = Network([x], f, var_scope="f")
        net2 = net1(name_scope="f2")
        func1 = NetworkFunction(net1["y"])
        func2 = NetworkFunction(net2["y"])
        self.assertListEqual(net1.variables, net2.variables, "net1/net2 variable equals check")
        with tf.name_scope("assigns"):
            reset_net1 = [tf.assign(v, tf.zeros_like(v)) for v in net1.variables]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            net1.set_session(sess)
            net2.set_session(sess)
            self.assertEquals(func1([[1.0]]), func2([[1.0]]))
            self.assertEquals(func1([[100.0]]), func2([[100.0]]))
            sess.run(reset_net1)
            self.assertEquals(func1([[456.0]]), [0])
            self.assertEquals(func2([[4567.0]]), [0])

