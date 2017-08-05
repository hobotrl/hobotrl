# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import hobotrl.network as network


class NNDistribution(object):
    """
    neural network backed probability distribution
    methods with postfix _run calculate corresponding operators to retrieve value
    """
    def __init__(self, **kwargs):
        super(NNDistribution, self).__init__()
        self._sess = None

    def entropy(self):
        """
        entropy of current distribution
        :return: operator calculating entropy, for entropy regularization
        """
        raise NotImplementedError()

    def entropy_run(self, inputs):
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

    def prob_run(self, inputs, sample):
        raise NotImplementedError()

    def log_prob(self):
        """
        logged probability(density) of sample
        :param sample:
        :return: operator calculating log_prob
        """
        raise NotImplementedError()

    def log_prob_run(self, inputs, input_sample):
        raise NotImplementedError()

    def sample(self):
        """
        :return: operator sampling from current distribution
        """
        raise NotImplementedError()

    def sample_run(self, inputs):
        raise NotImplementedError()

    def set_session(self, sess):
        self._sess = sess


class DiscreteDistribution(NNDistribution):

    def __init__(self, dist_function, input_sample, epsilon=1e-2, **kwargs):
        """
        :param dist_function Function returning probability distribution value
        :type dist_function: network.NetworkFunction
        :param input_sample: sample input calculating P(sample)
        :type input_sample: tf.Tensor or tf.placeholder
        :param epsilon: small positive offset to avoid zero probability.
        :param kwargs:
        """
        super(DiscreteDistribution, self).__init__(**kwargs)
        self._dist_function, self._epsilon = dist_function, epsilon
        self._input_sample = input_sample
        dist_n = dist_function.output().op.shape.as_list()[-1]
        self._dist_n = dist_n
        with tf.variable_scope("dist") as vs:
            net_dist = self._dist_function.output().op
            net_dist = net_dist + epsilon
            net_dist = net_dist / tf.reduce_sum(net_dist, axis=1, keep_dims=True)
            self._op_dist = net_dist
            log_dist = tf.log(self._op_dist)
            self._op_entropy = -tf.reduce_sum(log_dist * self._op_dist, axis=1)
            onehot_sample = tf.one_hot(self._input_sample, dist_n, dtype=tf.float32)
            self._op_prob = tf.reduce_sum(onehot_sample * self._op_dist, axis=1)
            self._op_log_prob = tf.reduce_sum(onehot_sample * log_dist, axis=1)

    def dist_input(self, inputs):
        if type(inputs) == list:
            feed_dict = self._dist_function.input_dict(*inputs)
        elif type(inputs) == dict:
            feed_dict = self._dist_function.input_dict(**inputs)
        else:
            feed_dict = self._dist_function.input_dict(inputs)
        return feed_dict

    def input_sample(self):
        return self._input_sample

    def dist_function(self):
        return self._dist_function

    def entropy(self):
        return self._op_entropy

    def entropy_run(self, inputs):
        return self._sess.run(self._op_entropy, feed_dict=self.dist_input(inputs))

    def dist(self):
        return self._op_dist

    def dist_run(self, inputs):
        return self._sess.run(self._op_dist, feed_dict=self.dist_input(inputs))

    def prob(self):
        return self._op_prob

    def prob_run(self, inputs, sample):
        feed_dict = self.dist_input(inputs)
        feed_dict.update({self._input_sample: sample})
        return self._sess.run(self._op_prob, feed_dict=feed_dict)

    def log_prob(self):
        return self._op_log_prob

    def log_prob_run(self, inputs, sample):
        feed_dict = self.dist_input(inputs)
        feed_dict.update({self._input_sample: sample})
        return self._sess.run(self._op_log_prob, feed_dict=feed_dict)

    def sample(self):
        # not implemented
        raise NotImplementedError()

    def sample_run(self, inputs):
        # distribution with shape [batch_size, dist_n]
        distribution = self._sess.run(self._op_dist, feed_dict=self.dist_input(inputs))
        sample_i = []
        for p in distribution:
            sample = np.random.choice(np.arange(self._dist_n), p=p)
            sample_i.append(sample)
        sample_i = np.asarray(sample_i)
        return sample_i


class NormalDistribution(NNDistribution):

    def __init__(self, dist_function, input_sample, epsilon=1e-2, **kwargs):
        """
        :param dist_function: NetworkFunction.
                output of function must be a dictionary of 2 vector:
                {"stddev": [batch_size, action_dim], "mean": [batch_size, action_dim]}
                where stdddev > 0 serving as std^2
        :type dist_function: network.NetworkFunction
        :param inputs_dist: list of distribution network input
        :param action_dim: integer indicating dimension of action
        :param input_sample: sample input placeholder, with shape: [batch_size] + action_shape
        :param epsilon: minimum stddev allowed, preventing premature convergence
        :param kwargs:
        """
        super(NormalDistribution, self).__init__(**kwargs)
        self._dist_function, self._epsilon, self._input_sample = dist_function, input_sample, epsilon
        self._action_dim = dist_function.output("mean").op.shape.as_list()[-1]
        with tf.variable_scope("dist") as vs:
            self._op_mean, self._op_stddev = dist_function.output("mean").op, dist_function.output("stddev").op
            self._op_stddev = self._op_stddev + epsilon
            self._op_entropy = (1 + tf.log(2 * np.pi * self._op_stddev)) / 2.0
            self._op_prob = 1.0 / tf.sqrt(2 * np.pi * self._op_stddev) \
                            * tf.exp(- tf.square(self._input_sample - self._op_mean) / (2.0 * self._op_stddev))

            self._op_log_prob = tf.log(self._op_prob)

    def dist_input(self, inputs):
        if type(inputs) == list:
            feed_dict = self._dist_function.input_dict(*inputs)
        elif type(inputs) == dict:
            feed_dict = self._dist_function.input_dict(**inputs)
        else:
            feed_dict = self._dist_function.input_dict(inputs)
        return feed_dict

    def entropy(self):
        return self._op_entropy

    def entropy_run(self, inputs):
        return self._sess.run(self._op_entropy, feed_dict=self.dist_input(inputs))

    def prob(self):
        return self._op_prob

    def prob_run(self, inputs, sample):
        feed_dict = self.dist_input(inputs)
        feed_dict[self._input_sample] = sample
        return self._sess.run(self._op_prob, feed_dict=feed_dict)

    def log_prob(self):
        return self._op_log_prob

    def log_prob_run(self, inputs, sample):
        feed_dict = self.dist_input(inputs)
        feed_dict[self._input_sample] = sample
        return self._sess.run(self._op_log_prob, feed_dict=feed_dict)

    def mean(self):
        return self._op_mean

    def mean_run(self, inputs):
        return self._sess.run(self._op_mean, feed_dict=self.dist_input(inputs))

    def stddev(self):
        return self._op_stddev

    def stddev_run(self, inputs):
        return self._sess.run(self._op_stddev, feed_dict=self.dist_input(inputs))

    def sample(self):
        # not implemented
        raise NotImplementedError()

    def sample_run(self, inputs):
        # distribution with shape [batch_size, dist_n]
        mean, stddev = self._sess.run(
            [self._op_mean, self._op_stddev],
            feed_dict=self.dist_input(inputs))
        sample_i = []
        stddev = np.sqrt(stddev)
        for i in range(len(mean)):
            mu, sigma = mean[i], stddev[i]
            sample_i.append(np.random.normal(mu, sigma))
        sample_i = np.asarray(sample_i)
        return sample_i


