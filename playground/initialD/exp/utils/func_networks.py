"""Network architectures.
We suggest to import architectures and feed hyper-params in using lambda
functions.

:author: Jingchu Liu
:date: Nov 28 2017
"""
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer


def f_dueling_q(inputs, num_actions, device_name="/gpu:0", l2=1e-2):
    inputs = inputs[0]
    inputs = inputs/128 - 1.0
    with tf.device(device_name):
        conv1 = layers.conv2d(
            inputs=inputs, filters=16, kernel_size=(8, 8), strides=1,
            kernel_regularizer=l2_regularizer(scale=l2),
            activation=tf.nn.relu, name='conv1')
        print conv1.shape
        pool1 = layers.max_pooling2d(
            inputs=conv1, pool_size=3, strides=4, name='pool1')
        print pool1.shape
        conv2 = layers.conv2d(
            inputs=pool1, filters=16, kernel_size=(5, 5), strides=1,
            kernel_regularizer=l2_regularizer(scale=l2),
            activation=tf.nn.relu, name='conv2')
        print conv2.shape
        pool2 = layers.max_pooling2d(
            inputs=conv2, pool_size=3, strides=3, name='pool2')
        print pool2.shape
        conv3 = layers.conv2d(
             inputs=pool2, filters=64, kernel_size=(3, 3), strides=1,
             kernel_regularizer=l2_regularizer(scale=l2),
             activation=tf.nn.relu, name='conv3')
        print conv3.shape
        pool3 = layers.max_pooling2d(
            inputs=conv3, pool_size=3, strides=2, name='pool3',)
        print pool3.shape
        depth = pool3.get_shape()[1:].num_elements()
        inputs = tf.reshape(pool3, shape=[-1, depth])
        print inputs.shape
        hid1 = layers.dense(
            inputs=inputs, units=256, activation=tf.nn.relu,
            kernel_regularizer=l2_regularizer(scale=l2), name='hid1')
        print hid1.shape
        hid2 = layers.dense(
            inputs=hid1, units=256, activation=tf.nn.relu,
            kernel_regularizer=l2_regularizer(scale=l2), name='hid2_adv')
        print hid2.shape
        adv = layers.dense(
            inputs=hid2, units=num_actions, activation=None,
            kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
            kernel_regularizer=l2_regularizer(scale=l2), name='adv')
        print adv.shape
        hid2 = layers.dense(
            inputs=hid1, units=256, activation=tf.nn.relu,
            kernel_regularizer=l2_regularizer(scale=l2), name='hid2_v')
        print hid2.shape
        v = layers.dense(
            inputs=hid2, units=1, activation=None,
            kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
            kernel_regularizer=l2_regularizer(scale=l2), name='v')
        print v.shape
        q = tf.add(adv, v, name='q')
        print q.shape

    return {"q": q}