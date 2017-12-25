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


def l2_distance(x_delta, y_delta, name=None):
    with ops.name_scope(name, "L2Distance", [x_delta, y_delta]) as name:
        g = tf.get_default_graph()
        with g.gradient_override_map({"Square": "L2DistanceSquareGrad",
                                      "Sqrt": "L2DistanceSqrtGrad",
                                      "Add": "L2DistanceAddGrad"}):
            return tf.sqrt(tf.square(y_delta) + tf.square(x_delta))


def l2_distance_weight(x1, y1, x2, y2, limit, name=None):
    return limit - l2_distance(x1 - x2, y1 - y2)


def trilinear_distance_weight(x1, y1, x2, y2, limit=2.0, name=None):
    with ops.name_scope(name, "TrilinearDistance", [x1, y1, x2, y2]) as name:
        return (limit - tf.abs(x1 - x2)) * (limit - tf.abs(y1 - y2)) / (2.0 * limit)


def l1_distance_weight(x1, y1, x2, y2, limit=2.0, name=None):
    with ops.name_scope(name, "L1Distance", [x1, y1, x2, y2]) as name:
        return ((limit - tf.abs(x1 - x2)) + (limit - tf.abs(y1 - y2))) / (2.0 * limit)


def l2_distance_weight_delta(x_delta, y_delta, limit, name=None):
    return limit - l2_distance(x_delta, y_delta)


def trilinear_distance_weight_delta(x_delta, y_delta, limit=2.0, name=None):
    with ops.name_scope(name, "TrilinearDistance", [x_delta, y_delta]) as name:
        return (limit - tf.abs(x_delta)) * (limit - tf.abs(y_delta)) / (2.0 * limit)


def l1_distance_weight_delta(x_delta, y_delta, limit=2.0, name=None):
    with ops.name_scope(name, "L1Distance", [x_delta, y_delta]) as name:
        return ((limit - tf.abs(x_delta)) + (limit - tf.abs(y_delta))) / (2.0 * limit)


class CoordUtil(object):

    @staticmethod
    def get_coord_tensor(n, h, w):
        x = tf.tile(tf.reshape(tf.range(w), shape=(1, 1, w, 1)), multiples=(n, h, 1, 1))
        y = tf.tile(tf.reshape(tf.range(h), shape=(1, h, 1, 1)), multiples=(n, 1, w, 1))
        return (y, x)

    @staticmethod
    def get_batch_index(n, h, w):
        return tf.tile(tf.reshape(tf.range(n), shape=(n, 1, 1, 1)), multiples=(1, h, w, 1))


def frame_trans(frame, move, kernel_size=3, name=None, weight_valid=False):
    # default kernel: 3x3
    with ops.name_scope(name, "FrameTrans", [frame, move, kernel_size]) as name:
        g = tf.get_default_graph()
    # with g.gradient_override_map({"FrameTrans": "FrameTransGrad"}):
        # frame: NHW3, move: NHW2
        frame_shape = tf.shape(frame, name="frame_shape")
        # n, h, w, c = tf.slice(frame_shape, 0, 1, name="frame_shape_n"), \
        #              tf.slice(frame_shape, 1, 1, name="frame_shape_h"), \
        #              tf.slice(frame_shape, 2, 1, name="frame_shape_w"), \
        #              tf.slice(frame_shape, 3, 1, name="frame_shape_c")
        n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
        y_const, x_const = CoordUtil.get_coord_tensor(n, h, w)
        y_deltas, x_deltas = tf.split(move, 2, axis=3)
        y_map, x_map = y_deltas + tf.to_float(y_const), x_deltas + tf.to_float(x_const)

        test_y_0 = tf.stop_gradient(tf.to_int32(y_map < 0.0))
        test_x_0 = tf.stop_gradient(tf.to_int32(x_map < 0.0))
        if kernel_size % 2 == 1:
            y_map_int = tf.to_int32(y_map + 0.5) * (1 - test_y_0) + tf.to_int32(y_map - 0.5) * test_y_0
            x_map_int = tf.to_int32(x_map + 0.5) * (1 - test_x_0) + tf.to_int32(x_map - 0.5) * test_x_0
        else:
            # y_map_int = tf.to_int32(y_map + 1.0) * (1 - test_y_0) + tf.to_int32(y_map) * test_y_0
            # x_map_int = tf.to_int32(x_map + 1.0) * (1 - test_x_0) + tf.to_int32(x_map) * test_x_0
            y_map_int = tf.to_int32(y_map) + (1 - test_y_0)
            x_map_int = tf.to_int32(x_map) + (1 - test_x_0)
        weights, activations = [], []

        max_distance = kernel_size / math.sqrt(2.0)
        batch_index = CoordUtil.get_batch_index(n, h, w)
        y_neighbors, x_neighbors = [], []
        y_valids, x_valids = [], []
        y_deltas, x_deltas = [], []
        for dy in range(kernel_size):
            y_neighbor = y_map_int + (dy - kernel_size / 2)
            y_neighbors.append(y_neighbor)
            y_valids.append(tf.logical_and(y_neighbor >= 0, y_neighbor < h))
            y_deltas.append(tf.to_float(y_neighbor) - y_map)
        for dx in range(kernel_size):
            x_neighbor = x_map_int + (dx - kernel_size / 2)
            x_neighbors.append(x_neighbor)
            x_valids.append(tf.logical_and(x_neighbor >= 0, x_neighbor < w))
            x_deltas.append(tf.to_float(x_neighbor) - x_map)
        for dx in range(kernel_size):
            for dy in range(kernel_size):
                y_neighbor, x_neighbor = y_neighbors[dy], x_neighbors[dx]
                y_valid = y_valids[dy]
                x_valid = x_valids[dx]
                y_delta, x_delta = y_deltas[dy], x_deltas[dx]
                valid = tf.logical_and(y_valid, x_valid)
                # logging.warning("y_neighbor, x_neighbor, y_valid, x_valid, y_map, x_map: %s, %s, %s, %s, %s, %s,", y_neighbor, x_neighbor, y_valid, x_valid, y_map, x_map)
                weights.append(
                    # l2_distance_weight_delta(x_delta, y_delta, max_distance) * tf.to_float(valid)
                    trilinear_distance_weight_delta(x_delta, y_delta, limit=kernel_size / 2.0) * tf.to_float(valid)
                    # l1_distance_weight_delta(x_delta, y_delta, limit=kernel_size / 2.0) * tf.to_float(valid)
                )
                # validate coordinates: invalid coordinates to [0, 0]
                y_neighbor = y_neighbor * tf.to_int32(y_valid)
                x_neighbor = x_neighbor * tf.to_int32(x_valid)
                nyx_coord = tf.concat((batch_index, y_neighbor, x_neighbor), axis=3)
                # activations.append(tf.stop_gradient(tf.gather_nd(frame, nyx_coord) * tf.to_float(valid)))
                activations.append(tf.gather_nd(frame, nyx_coord))
                if weight_valid:
                    weights[-1] = weights[-1] * tf.to_float(valid)
                else:
                    # actionvation valid
                    activations[-1] = activations[-1] * tf.to_float(valid)
                # logging.warning("weight:%s, activation:%s", weights[-1], activations[-1])
        # reweight
        sum_weights = tf.add(tf.add_n(weights), 1e-7, name="sum_weight")
        # logging.warning("sum_weight:%s", sum_weights)
        output = tf.add_n([activations[i] * weights[i] for i in range(len(weights))], name="next_frame")
        output = tf.div(output, sum_weights, name="next_frame_normed")
        return output
