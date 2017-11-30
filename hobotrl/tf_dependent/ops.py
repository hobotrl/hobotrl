# -*- coding: utf-8 -*-

import logging
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops


@ops.RegisterGradient("ATanhGrad")
def atanh_grad(op, grad):
    x = op.inputs[0]
    y = op.outputs[0]
    org_grad = gen_math_ops._tanh_grad(y, grad)
    # return [gen_math_ops._abs(gen_math_ops.sign(y) - gen_math_ops.sign(grad)) / 2 * org_grad +
    #         gen_math_ops._abs(gen_math_ops.sign(y) + gen_math_ops.sign(grad)) / 2 * grad]
    return [gen_math_ops._abs(gen_math_ops.sign(y) - gen_math_ops.sign(grad)) * org_grad +
            gen_math_ops._abs(gen_math_ops.sign(y) + gen_math_ops.sign(grad)) / 2 * grad]


def atanh(x, name=None):
    # tf.RegisterGradient("AtanhGrad")(atanh_grad)
    with ops.name_scope(name, "ATanh", [x]) as name:
        g = tf.get_default_graph()
        with g.gradient_override_map({"Tanh": "ATanhGrad"}):
            return tf.tanh(x)


@ops.RegisterGradient("L2DistanceSqrtGrad")
def l2_distance_sqrt_grad(op, grad):
    return [grad]


@ops.RegisterGradient("L2DistanceAddGrad")
def l2_distance_add_grad(op, grad):
    epsilon = 1e-6
    x, y = op.inputs[0], op.inputs[1]
    wx, wy = x + epsilon, y + epsilon
    ws = wx + wy
    return [grad * (wx / ws), grad * (wy / ws)]


@ops.RegisterGradient("L2DistanceSquareGrad")
def l2_distance_square_grad(op, grad):
    x = op.inputs[0]
    return [tf.sign(x) * grad]


def l2_distance(x1, y1, x2, y2, name=None):
    with ops.name_scope(name, "L2Distance", [x1, y1, x2, y2]) as name:
        g = tf.get_default_graph()
        with g.gradient_override_map({"Square": "L2DistanceSquareGrad",
                                      "Sqrt": "L2DistanceSqrtGrad",
                                      "Add": "L2DistanceAddGrad"}):
            return tf.sqrt(tf.square(x1 - x2) + tf.square(y1 - y2))


class CoordUtil(object):

    @staticmethod
    def get_coord_tensor(n, h, w):
        x = tf.tile(tf.reshape(tf.range(w), shape=(1, 1, w, 1)), multiples=(n, h, 1, 1))
        y = tf.tile(tf.reshape(tf.range(h), shape=(1, h, 1, 1)), multiples=(n, 1, w, 1))
        return (y, x)

    @staticmethod
    def get_batch_index(n, h, w):
        return tf.tile(tf.reshape(tf.range(n), shape=(n, 1, 1, 1)), multiples=(1, h, w, 1))


def frame_trans(frame, move, kernel_size=3, name=None):
    # default kernel: 3x3
    with ops.name_scope(name, "FrameTrans", [frame, move, kernel_size]) as name:
        g = tf.get_default_graph()
    # with g.gradient_override_map({"FrameTrans": "FrameTransGrad"}):
        # frame: NHW3, move: NHW2
        frame_shape = tf.shape(frame)
        n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
        y_const, x_const = CoordUtil.get_coord_tensor(n, h, w)
        y_delta, x_delta = tf.split(move, 2, axis=3)
        y_map, x_map = y_delta + tf.to_float(y_const), x_delta + tf.to_float(x_const)

        test = tf.stop_gradient(tf.to_int32(y_map < 0.0))
        y_map_int = tf.to_int32(y_map + 0.5) * (1 - test) + tf.to_int32(y_map - 0.5) * test
        test = tf.stop_gradient(tf.to_int32(x_map < 0.0))
        x_map_int = tf.to_int32(x_map + 0.5) * (1 - test) + tf.to_int32(x_map - 0.5) * test
        #
        # h, w = tf.stop_gradient(h), tf.stop_gradient(w)
        # test = tf.stop_gradient(tf.to_float(y_map < 0.0))
        # y_map = y_map * (1 - test)
        # test = tf.stop_gradient(tf.to_float(y_map > (tf.to_float(h) - 1.0)))
        # y_map = y_map * (1 - test) + (tf.to_float(h) - 1.0) * test
        # test = tf.stop_gradient(tf.to_float(x_map < 0.0))
        # x_map = x_map * (1 - test)
        # test = tf.stop_gradient(tf.to_float(x_map > (tf.to_float(w) - 1.0)))
        # x_map = x_map * (1 - test) + (tf.to_float(w) - 1.0) * test

        # y_map_int, x_map_int = tf.to_int32(y_map + 0.5), tf.to_int32(x_map + 0.5)
        # y_map_int, x_map_int = tf.stop_gradient(y_map_int), tf.stop_gradient(x_map_int)
        weights, activations = [], []
        # with tf.control_dependencies([tf.assert_positive(
        #     tf.to_float(tf.logical_not(
        #         tf.logical_or(tf.is_nan(y_map), tf.is_nan(x_map))
        #     ))
        # )]):
        #     with tf.control_dependencies([tf.assert_less_equal(
        #         tf.sqrt(tf.square(tf.to_float(y_map_int)-y_map) + tf.square(tf.to_float(x_map_int) - x_map)),
        #                     math.sqrt(2.0)/2)]):

        max_distance = kernel_size / math.sqrt(2.0)
        batch_index = CoordUtil.get_batch_index(n, h, w)
        for dx in range(kernel_size):
            for dy in range(kernel_size):
                y_neighbor, x_neighbor = y_map_int + (dy - kernel_size / 2), x_map_int + (dx - kernel_size / 2)
                y_valid = tf.logical_and(y_neighbor >= 0, y_neighbor < h)
                x_valid = tf.logical_and(x_neighbor >= 0, x_neighbor < w)
                valid = tf.logical_and(y_valid, x_valid)
                # logging.warning("y_neighbor, x_neighbor, y_valid, x_valid, y_map, x_map: %s, %s, %s, %s, %s, %s,", y_neighbor, x_neighbor, y_valid, x_valid, y_map, x_map)
                weights.append(
                    tf.subtract(
                        tf.to_float(max_distance),
                        l2_distance(tf.to_float(x_neighbor), tf.to_float(y_neighbor), x_map, y_map)
                    , name="weight")
                )
                # with tf.control_dependencies([tf.assert_greater_equal(weights[-1], -0.0)]):
                # validate coordinates
                y_neighbor = y_neighbor * tf.to_int32(y_valid)
                x_neighbor = x_neighbor * tf.to_int32(x_valid)
                nyx_coord = tf.concat((batch_index, y_neighbor, x_neighbor), axis=3)
                # activations.append(tf.stop_gradient(tf.gather_nd(frame, nyx_coord) * tf.to_float(valid)))
                activations.append(tf.gather_nd(frame, nyx_coord) * tf.to_float(valid))
                # logging.warning("weight:%s, activation:%s", weights[-1], activations[-1])
        # reweight
        sum_weights = tf.add(tf.add_n(weights), 1e-7, name="sum_weight")
        # logging.warning("sum_weight:%s", sum_weights)
        output = tf.add_n([activations[i] * weights[i] for i in range(len(weights))], name="next_frame")
        output = tf.div(output, sum_weights, name="next_frame_normed")
        return output
