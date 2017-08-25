# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl

import hobotrl.network as network
import hobotrl.sampling as sampling
import hobotrl.target_estimate as target_estimate
import hobotrl.tf_dependent.distribution as distribution
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.policy import StochasticPolicy
from value_based import GreedyStateValueFunction


class ActorCriticUpdater(network.NetworkUpdater):
    def __init__(self, policy_dist, v_function, target_estimator, entropy=1e-3, actor_weight=1.0):
        """
        Actor Critic methods, for both continuous and discrete action spaces.

        :param policy_dist:
        :type policy_dist: distribution.NNDistribution
        :param v_function: Function calculating state value
        :type v_function: network.NetworkFunction
        :param target_estimator:
        :type target_estimator:
        :param num_actions:
        """
        super(ActorCriticUpdater, self).__init__()
        self._policy_dist, self._v_function = policy_dist, v_function
        self._target_estimator = target_estimator
        self._entropy = entropy
        with tf.name_scope("ActorCriticUpdater"):
            with tf.name_scope("input"):
                self._input_target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_v")
                self._input_action = policy_dist.input_sample()
                self._input_entropy = tf.placeholder(dtype=tf.float32, shape=[], name="input_entropy")
            op_v = v_function.output().op
            with tf.name_scope("value"):
                td = self._input_target_v - op_v
                self._q_loss = tf.reduce_mean(network.Utils.clipped_square(td))
            with tf.name_scope("policy"):
                advantage = self._input_target_v - op_v
                self._advantage = advantage
                _mean, _var = tf.nn.moments(advantage, axes=[0])
                self._std_advantage = advantage / (tf.sqrt(_var) + 1.0)
                # self._std_advantage = self._advantage
                pi_loss = tf.reduce_mean(self._policy_dist.log_prob() * tf.stop_gradient(self._std_advantage))
                entropy_loss = tf.reduce_mean(self._input_entropy * self._policy_dist.entropy())
                self._pi_loss = pi_loss
            self._op_loss = self._q_loss - actor_weight * (self._pi_loss + entropy_loss)
            print "advantage, self._policy_dist.entropy(), self._policy_dist.log_prob()", advantage, self._policy_dist.entropy(), self._policy_dist.log_prob()
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._v_function.variables +
                                                               self._policy_dist._dist_function.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done = batch["state"], \
                                                          batch["action"], \
                                                          batch["reward"], \
                                                          batch["next_state"], \
                                                          batch["episode_done"]
        target_value = self._target_estimator.estimate(state, action, reward, next_state, episode_done)
        feed_dict = self._v_function.input_dict(state)
        feed_dict.update(self._policy_dist.dist_function().input_dict(state))
        feed_more = {
            self._input_action: action,
            self._input_target_v: target_value,
            self._input_entropy: self._entropy
        }
        feed_dict.update(feed_more)
        fetch_dict = {
            "advantage": self._advantage,
            "std_advantage": self._std_advantage,
            "target_value": target_value,
            "pi_loss": self._pi_loss,
            "q_loss": self._q_loss,
            "entropy": self._policy_dist.entropy(),
            "log_prob": self._policy_dist.log_prob(),
        }
        if isinstance(self._policy_dist, hrl.tf_dependent.distribution.NormalDistribution):
            fetch_dict.update({
                "stddev": self._policy_dist.stddev(),
                "mean": self._policy_dist.mean()
            })
        else:
            pass
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)


class ForwardUpdater(network.NetworkUpdater):
    def __init__(self, forward_function, feature_function, policy_dist):
        super(ForwardUpdater, self).__init__()
        self._forward_function, self._feature_function, self._policy_dist = \
            forward_function, feature_function, policy_dist

        with tf.name_scope("ForwardUpdater"):
            op_phi_next_state_hat = forward_function.output().op
            op_phi_next_state = feature_function.output().op

            # forward loss calculation
            with tf.name_scope("forward"):
                forward_loss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(op_phi_next_state_hat, op_phi_next_state)),
                                                    name="forward_loss")
                self._forward_loss = forward_loss
            self._op_loss = self._forward_loss

        self._update_operation = network.MinimizeLoss(self._op_loss, var_list=self._forward_function.variables +
                                                                              self._feature_function.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done = batch["state"], \
                                                          batch["action"], \
                                                          batch["reward"], \
                                                          batch["next_state"], \
                                                          batch["episode_done"]
        feed_dict = self._feature_function.input_dict(state)
        feed_dict.update(self._feature_function.input_dict(next_state))
        feed_dict.update(self._forward_function.input_dict(action))

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"forward loss": self._forward_loss})


class InverseUpdater(network.NetworkUpdater):
    def __init__(self, inverse_function, feature_function, policy_dist):
        super(InverseUpdater, self).__init__()
        self._inverse_function, self._feature_function, self._policy_dist = \
            inverse_function, feature_function, policy_dist

        with tf.name_scope("InverseUpdater"):
            with tf.name_scope("input"):
                self._input_action = policy_dist.input_sample()

            op_action_hat = inverse_function.output().op

            # inverse loss calculation
            with tf.name_scope("inverse"):
                inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._input_action,
                                                                                      logits=op_action_hat))
                self._inverse_loss = inverse_loss

            self._op_loss = self._inverse_loss

        self._update_operation = network.MinimizeLoss(self._op_loss, var_list=self._inverse_function.variables +
                                        self._feature_function.variables + self._policy_dist._dist_function.variables)
    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done = batch["state"], \
                                                          batch["action"], \
                                                          batch["reward"], \
                                                          batch["next_state"], \
                                                          batch["episode_done"]
        feed_dict = self._feature_function.input_dict(state)
        feed_dict.update(self._feature_function.input_dict(next_state))
        feed_dict.update(self._policy_dist.dist_function().input_dict(state))
        feed_more = {self._input_action: action}
        feed_dict.update(feed_more)

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"inverse loss": self._inverse_loss})