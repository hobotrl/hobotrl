# -*- coding: utf-8 -*-
# Reinforcement Learning with Deep Energy-Based Policies https://arxiv.org/abs/1702.08165
# https://github.com/haarnoja/softqlearning

import sys
import logging
import tensorflow as tf
import numpy as np
from hobotrl import network

from hobotrl.core import Policy
from hobotrl.network import Network, NetworkUpdater, NetworkFunction, MinimizeLoss, UpdateRun, LocalOptimizer


class SVGDUpdater(NetworkUpdater):
    def __init__(self, generator_net, logprob_net, m_particles=16):
        """
        SVGD, updates generator_net to match PDF of logprob_net.
         Using unit gaussian kernel.
        :param generator_net: generator_net(state, noise) => action
        :type generator_net: Network
        :param logprob_net: logprob_net(state, action) => log pdf of action
        :type logprob_net: Network
        """
        super(SVGDUpdater, self).__init__()
        self._m_particles = m_particles
        m = m_particles
        self._generator_net, self._logprob_net = generator_net, logprob_net
        state_shape = generator_net.inputs[0].shape.as_list()
        noise_shape = generator_net.inputs[1].shape.as_list()
        action_shape = logprob_net.inputs[1].shape.as_list()
        dim_state = state_shape[1]
        self._dim_noise = noise_shape[1]
        dim_action = action_shape[1]
        with tf.name_scope("inputs"):
            self._input_state = tf.placeholder(tf.float32, state_shape, "input_state")
            self._input_noise = tf.placeholder(tf.float32, noise_shape, "input_noise")
        with tf.name_scope("generate"):
            state_batch = tf.shape(self._input_state)[0]
            state = tf.reshape(tf.tile(
                tf.reshape(self._input_state, shape=[-1, 1, dim_state])
                , (1, m, 1))
                , shape=[-1, dim_state])
            noise = tf.tile(self._input_noise, (state_batch, 1))
            # generate action with tuple:
            #   (s0, n0), (s0, n1), ..., (s1, n0), (s1, n1), ...
            generator_net = generator_net([state, noise], name_scope="batch_generator")
            for name in generator_net.outputs:
                actions = generator_net[name].op
                break
            # actions: [bs * m, dim_a]
            action_square = tf.tile(tf.reshape(actions, [-1, 1, m, dim_action]), (1, m, 1, 1))
            # sub: [b_s, m, m, dim_a]
            action_sub = tf.transpose(action_square, perm=[0, 2, 1, 3]) - action_square
            # dis square: [b_s, m, m]
            dis_square = tf.reduce_sum(tf.square(action_sub), axis=3)
            # h: [b_s]
            h = tf.reduce_mean(tf.sqrt(dis_square), axis=(1, 2))
            h = h / (2 * np.log(m + 1))
            # k: [bs, m, m]
            k = tf.exp(-1.0 / tf.reshape(h, (-1, 1, 1)) * dis_square)
            # dk: [bs, m, m, dim_a]
            dk = tf.reshape(k, (-1, m, m, 1)) * (2 / tf.reshape(h, (-1, 1, 1, 1))) * action_sub
            # dlogprob: [bs, m, 1]
            logprob_net = logprob_net([state, actions], name_scope="batch_logprob")
            for name in logprob_net.outputs:
                action_logprob = logprob_net[name].op
                break
            dlogp = tf.gradients(action_logprob, actions)
            # dlogp/da: [bs, m, m, dim_a]
            dlogp_matrix = tf.tile(tf.reshape(dlogp, (state_batch, 1, m, dim_action)), (1, m, 1, 1))
            # svgd gradient: [bs, m, m, dim_a]
            grad_svgd = tf.reshape(k, (-1, m, m, 1)) * dlogp_matrix + dk
            # [bs, m, dim_a]
            grad_svgd = tf.reduce_mean(grad_svgd, axis=2)
            # [bs * m, dim_a]
            grad_svgd = tf.reshape(grad_svgd, (-1, dim_action))
            generator_loss = tf.reduce_mean(-tf.stop_gradient(grad_svgd) * actions)
            self._loss = generator_loss
            self._grad_svgd = grad_svgd
            self._op = MinimizeLoss(self._loss, var_list=generator_net.variables)

    def declare_update(self):
        return self._op

    def update(self, sess, batch, *args, **kwargs):
        state = batch["state"]
        noise = np.random.normal(0, 1, (self._m_particles, self._dim_noise))
        return UpdateRun(feed_dict={self._input_state: state,
                                    self._input_noise: noise})


class SoftStochasticPolicy(Policy):
    def __init__(self, ):
        super(SoftStochasticPolicy, self).__init__()

    def act(self, state, **kwargs):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dim_action = 2
    dim_noise = 2
    dim_state = 4

    def f_generator(inputs):
        state, noise = inputs[0], inputs[1]
        se = tf.concat([state, noise], axis=1)
        action = network.Utils.layer_fcs(se, [256, 256], dim_action, activation_out=None, var_scope="action")
        return {"action": action}

    def f_normal(inputs):
        state, action = inputs[0], inputs[1]
        se = tf.concat([state, action], axis=1)
        mean = np.array([[0.5, 0.5]])
        return {"logp": - tf.reduce_sum(tf.square(action - mean), axis=1)}

    def f_normal2(inputs):
        action = inputs[1]
        mean1 = np.array([[1.0, 1.0]])
        mean2 = np.array([[-1.0, -1.0]])
        n1 = tf.exp(-tf.reduce_sum(tf.square(action - mean1), axis=1))
        n2 = tf.exp(-tf.reduce_sum(tf.square(action - mean2), axis=1))
        logp = tf.log(n1 + n2)
        return {"logp": logp}
    
    f_q = f_normal2
    input_state = tf.placeholder(dtype=tf.float32, shape=[None, dim_state], name="input_state")
    input_action = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name="input_action")
    input_noise = tf.placeholder(dtype=tf.float32, shape=[None, dim_noise], name="input_noise")
    generator_net = Network([input_state, input_noise], f_generator, var_scope="generator")
    generator_func = NetworkFunction(generator_net["action"])
    logprob_net = Network([input_state, input_action], f_q, var_scope="q")
    optimizer = LocalOptimizer()
    batch_size = 8
    m_particles = 128
    test_batch = 256
    optimizer.add_updater(SVGDUpdater(generator_net, logprob_net, m_particles=m_particles), name="svgd")
    optimizer.compile()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    generator_func.set_session(sess)
    for i in range(1000):
        state = np.random.normal(0, 1, (batch_size, dim_state))
        optimizer.update("svgd", sess, {"state": state})
        info = optimizer.optimize_step(sess)
        if i % 10 == 0:
            state = np.random.normal(0, 1, (1, dim_state))
            state = np.repeat(state, test_batch, axis=0)
            noise = np.random.normal(0, 1, (test_batch, dim_noise))
            actions = generator_func(state, noise)
            coord = np.transpose(actions)
            plt.plot(coord[0], coord[1], 'ro')
            plt.axis([-2, 2, -2, 2])
            plt.show()

