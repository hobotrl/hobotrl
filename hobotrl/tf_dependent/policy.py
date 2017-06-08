#
# -*- coding: utf-8 -*-

import logging

import tensorflow as tf
import numpy as np

import hobotrl as hrl
from distribution import DiscreteDistribution
from mixin import NNStochasticPolicyMixin


class DiscreteNNPolicy(NNStochasticPolicyMixin):

    def __init__(self, state_shape, num_actions, f_create_net,
                 training_params=None, entropy=0.01, gamma=0.9, train_interval=8, **kwargs):
        """

        :param state_shape:
        :param num_actions:
        :param f_create_net:
        :param training_params: tuple containing training parameters.
            two members:
                optimizer_td : Tensorflow optimizer for gradient-based opt.
                target_sync_rate: for other use.
        :param entropy: entropy regularization term
        :param gamma: reward discount factor
        :param train_interval: policy update interval
        :param kwargs:
        """
        kwargs.update({
            "state_shape": state_shape,
            "num_actions": num_actions,
            "training_params": training_params,
            "entropy": entropy,
            "gamma": gamma,
            "train_interval": train_interval
        })
        super(DiscreteNNPolicy, self).__init__(**kwargs)
        self.state_shape, self.num_actions = state_shape, num_actions
        self.entropy, self.reward_decay, self.train_interval = entropy, gamma, train_interval
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="input_state")
        self.input_action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="input_action")
        self.input_advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_advantage")
        self.input_entropy = tf.placeholder(dtype=tf.float32, name="input_entropy")
        with tf.variable_scope("policy") as vs:
            self.distribution = DiscreteDistribution(f_create_net, [self.input_state], num_actions,
                                                     input_sample=self.input_action, **kwargs)
        vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
        self.episode_buffer = hrl.playback.MapPlayback(train_interval, {
            "s": state_shape,
            "a": [1],
            "r": [1],
            "t": [1],
            "s1": state_shape
        }, pop_policy="sequence")

        # other operators for training
        self.op_entropy = tf.reduce_mean(self.distribution.entropy())
        self.pi_loss = tf.reduce_mean(self.distribution.log_prob() * self.input_advantage) \
                       + self.input_entropy * self.op_entropy

        if training_params is None:
            optimizer = tf.train.AdamOptimizer()
        else:
            optimizer = training_params[0]
        self.op_train = optimizer.minimize(self.pi_loss, var_list=vars_policy)
        # self.pi_loss = self.pi_loss + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), "policy")
        self.train_countdown = self.train_interval
        self.sess = None

    def act(self, state, **kwargs):
        return self.distribution.sample_run(self.sess, [np.asarray([state])])[0]

    def update_policy(self, state, action, reward, next_state, episode_done, **kwargs):
        self.episode_buffer.push_sample(
            {
                's': state,
                'a': action,
                's1': next_state,
                'r': np.asarray([reward], dtype=float),
                't': np.asarray([episode_done], dtype=float)
            }, reward
        )
        self.train_countdown -= 1
        if episode_done or self.train_countdown == 0:
            self.train_countdown = self.train_interval
            batch_size = self.episode_buffer.get_count()
            batch = self.episode_buffer.sample_batch(batch_size)
            self.episode_buffer.reset()
            Si, Ai, Ri, Sj, T = np.asarray(batch['s'], dtype=float), batch['a'], batch['r'], \
                                np.asarray(batch['s1'], dtype=float), batch['t']
            # computing advantage
            R = np.zeros(shape=[batch_size, 1], dtype=float)
            V = self.get_value(state=Si, sess=self.sess)
            V = np.max(V, axis=-1, keepdims=True)
            if episode_done:
                r = 0.0
            else:
                last_v = self.get_value(state=np.asarray([next_state]), sess=self.sess)
                r = np.max(last_v)

            for i in range(batch_size):
                index = batch_size - i - 1
                if T[index][0] != 0:
                    # logging.warning("Terminated!, Ri:%s, Vi:%s", Ri[index], Vi[index])
                    r = 0
                r = Ri[index][0] + self.reward_decay * r
                R[index][0] = r

            target_name = "Ternimate" if episode_done else "bootstrap"
            logging.warning("Target from %s: [ %s ... %s]", target_name, R[0], R[-1])

            advantage = R - V
            _, loss, entropy = self.sess.run([self.op_train, self.pi_loss, self.op_entropy], feed_dict={self.input_state: Si,
                                                      self.input_action: Ai,
                                                      self.input_entropy: self.entropy,
                                                      self.input_advantage: advantage})

            return {"policy_loss": loss, "entropy": entropy, "advantage": advantage, "V": V}

        return {}
