import sys
sys.path.append('../../../')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.tf_dependent.distribution import DiscreteDistribution


def f_net(inputs, num_outputs):
    inputs = inputs[0]
    depth_state = inputs.get_shape()[1:].num_elements()
    inputs = tf.reshape(inputs, shape=[-1, depth_state], name='inputs')

    dist = layers.dense(
        inputs=inputs, units=num_outputs,
        activation=tf.nn.softmax,
        trainable=True
    )

    return dist

inputs_dist = tf.placeholder(shape=[None, 3], dtype=tf.float32)
input_sample = tf.placeholder(shape=[None,], dtype=tf.int32)

dist_n = 5
batch_size = 10

dd = DiscreteDistribution(
    f_create_net=f_net,
    inputs_dist=[inputs_dist],
    dist_n=dist_n,
    input_sample=input_sample,
    epsilon=1e-3
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print "="*20
print "Test entropy_run:"
print dd.entropy_run(
    sess,
    [np.random.rand(batch_size, 3)]
)

print "="*20
print "Test prob_run:"
print dd.prob_run(
    sess,
    [np.random.rand(batch_size, 3)],
    np.random.randint(0, dist_n, [batch_size])
)

print "="*20
print "Test sample_run:"
print dd.sample_run(
    sess,
    [np.random.rand(batch_size, 3)]
)

print "="*20
print "Test dist_run:"
print dd.dist_run(
    sess,
    [np.random.rand(batch_size, 3)]
)
