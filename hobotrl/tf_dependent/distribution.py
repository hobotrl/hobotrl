#
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class NNDistribution(object):
    """
    neural network backed probability distribution
    methods with postfix _run calculate corresponding operators to retrieve value
    """
    def __init__(self, **kwargs):
        super(NNDistribution, self).__init__()

    def entropy(self):
        """
        entropy of current distribution
        :return: operator calculating entropy, for entropy regularization
        """
        raise NotImplementedError()

    def entropy_run(self, sess, inputs):
        """
        calculate entropy, according to inputs
        :param sess: tf Session
        :param inputs: list of input value
        :return:
        """
        raise NotImplementedError()

    def prob(self):
        """
        calculate probability/probability density of sample w.r.t. current distribution
        :param sample:
        :return: operator calculating prob
        """
        raise NotImplementedError()

    def prob_run(self, sess, inputs, sample):
        raise NotImplementedError()

    def log_prob(self):
        """
        logged probability(density) of sample
        :param sample:
        :return: operator calculating log_prob
        """
        raise NotImplementedError()

    def log_prob_run(self, sess, inputs, input_sample):
        raise NotImplementedError()

    def sample(self):
        """
        :return: operator sampling from current distribution
        """
        raise NotImplementedError()

    def sample_run(self, sess, inputs):
        raise NotImplementedError()


class DiscreteDistribution(NNDistribution):

    def __init__(self, f_create_net, inputs_dist, dist_n, input_sample, epsilon=1e-2, **kwargs):
        """
        :param f_create_net: function to create network.
                output of network must be a 1-normalized distribution over `dist_n` categories: [batch_size, dist_n]
        :param inputs_dist: list of distribution network input
        :param dist_n: count of categories
        :param input_sample: sample input placeholder, with shape: [batch_size, 1]
        :param kwargs:
        """
        super(DiscreteDistribution, self).__init__(**kwargs)
        self.epsilon = epsilon
        with tf.variable_scope("dist") as vs:
            self.inputs_dist, self.input_sample, self.dist_n = inputs_dist, input_sample, dist_n
            net_dist = f_create_net(inputs_dist, dist_n)
            net_dist = net_dist + epsilon
            net_dist = net_dist / tf.reduce_sum(net_dist, axis=1, keep_dims=True)
            self.net_dist = net_dist
            log_dist = tf.log(self.net_dist)
            self.net_entropy = -tf.reduce_sum(log_dist * self.net_dist, axis=1, keep_dims=True)
            onehot_sample = tf.reshape(tf.one_hot(input_sample, dist_n, dtype=tf.float32), [-1, dist_n])
            self.net_prob = tf.reduce_sum(onehot_sample * self.net_dist, axis=1, keep_dims=True)
            self.net_log_prob = tf.reduce_sum(onehot_sample * log_dist, axis=1, keep_dims=True)
            print "shapes:", self.inputs_dist, self.input_sample, self.dist_n, self.net_dist, self.net_entropy, \
                onehot_sample, self.net_prob, self.net_log_prob

    def entropy(self):
        return self.net_entropy

    def entropy_run(self, sess, inputs):
        return sess.run([self.net_entropy], feed_dict=dict(zip(self.inputs_dist, inputs)))[0]

    def prob(self):
        return self.net_prob

    def prob_run(self, sess, inputs, sample):
        return sess.run([self.net_prob],
                        feed_dict=dict(zip(self.inputs_dist, inputs) + [self.input_sample, sample]))[0]

    def log_prob(self):
        return self.net_log_prob

    def log_prob_run(self, sess, inputs, sample):
        return sess.run([self.net_log_prob],
                        feed_dict=dict(zip(self.inputs_dist, inputs) + [self.input_sample, sample]))[0]

    def sample(self):
        # not implemented
        raise NotImplementedError()

    def sample_run(self, sess, inputs):
        # distribution with shape [batch_size, dist_n]
        distribution = sess.run([self.net_dist], feed_dict=dict(zip(self.inputs_dist, inputs)))[0]
        sample_i = []
        for p in distribution:
            sample_i.append(np.random.choice(np.arange(self.dist_n), p=p))
        sample_i = np.asarray(sample_i)
        return sample_i
