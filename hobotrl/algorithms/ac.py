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


class DiscreteActorCriticUpdater(network.NetworkUpdater):
    def __init__(self, policy_dist, q_function, target_estimator, entropy=1e-3, max_advantage=10.0, actor_weight=0.1):
        """
        :param policy_dist:
        :type policy_dist: distribution.DiscreteDistribution
        :param q_function:
        :type q_function: network.NetworkFunction
        :param target_estimator:
        :type target_estimator: target_estimate.TargetEstimator
        :param num_actions:
        """
        super(DiscreteActorCriticUpdater, self).__init__()
        self._policy_dist, self._q_function = policy_dist, q_function
        self._target_estimator = target_estimator
        self._entropy = entropy
        self._num_actions = q_function.output().op.shape.as_list()[-1]
        with tf.name_scope("DiscreteActorCriticUpdate"):
            with tf.name_scope("input"):
                self._input_target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_q")
                self._input_action = policy_dist.input_sample()
                self._input_entropy = tf.placeholder(dtype=tf.float32, shape=[], name="input_entropy")
            op_q = q_function.output().op
            with tf.name_scope("value"):
                selected_q = tf.reduce_sum(tf.one_hot(self._input_action, self._num_actions, dtype=tf.float32) * op_q, axis=1)
                td = self._input_target_q - selected_q
                self._q_loss = tf.reduce_mean(network.Utils.clipped_square(td))
            with tf.name_scope("policy"):
                # v = tf.reduce_max(op_q, axis=1)  # state value for greedy policy
                v = tf.reduce_sum(op_q * policy_dist.dist(), axis=1)  # real state value for actual policy
                advantage = self._input_target_q - v
                advantage = tf.clip_by_value(advantage, -max_advantage, max_advantage, name="advantage")
                pi_loss = tf.reduce_mean(self._policy_dist.log_prob() * tf.stop_gradient(advantage))
                entropy_loss = tf.reduce_mean(self._input_entropy * self._policy_dist.entropy())
                self._pi_loss = pi_loss
            # self._op_loss = self._q_loss - actor_weight * (self._pi_loss + entropy_loss)
            self._op_loss = self._q_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._q_function.variables +
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
        feed_dict = self._q_function.input_dict(state)
        feed_dict.update(self._policy_dist.dist_function().input_dict(state))
        feed_dict.update({
            self._input_action: action,
            self._input_target_q: target_value,
            self._input_entropy: self._entropy
        })
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={
            "target_value": target_value,
            "pi_loss": self._pi_loss,
            "q_loss": self._q_loss,
            "entropy": self._policy_dist.entropy()
        })

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
            # self._op_loss = self._q_loss - (self._pi_loss + entropy_loss)
            self._op_loss = self._q_loss
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


class DiscretePGUpdater(network.NetworkUpdater):
    def __init__(self, policy_dist, v_function, target_estimator, entropy=1e-3, max_advantage=10.0):
        """
        :param policy_dist:
        :type policy_dist: distribution.DiscreteDistribution
        :param v_function:
        :type v_function: network.NetworkFunction
        :param target_estimator:
        :type target_estimator:
        """
        super(DiscretePGUpdater, self).__init__()
        self._policy_dist, self._v_function = policy_dist, v_function
        self._target_estimator = target_estimator
        self._entropy = entropy
        self._num_actions = v_function.output().op.shape.as_list()[-1]
        with tf.name_scope("DiscreteActorCriticUpdate"):
            with tf.name_scope("input"):
                self._input_target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_q")
                self._input_action = policy_dist.input_sample()
                self._input_entropy = tf.placeholder(dtype=tf.float32, shape=[], name="input_entropy")
            op_v = v_function.output().op
            with tf.name_scope("policy"):
                advantage = self._input_target_v - op_v
                advantage = tf.clip_by_value(advantage, -max_advantage, max_advantage, name="advantage")
                self._pi_loss = tf.reduce_mean(self._policy_dist.log_prob() * tf.stop_gradient(advantage))
                entropy_loss = self._input_entropy * tf.reduce_mean(self._policy_dist.entropy())
            self._op_loss = - (self._pi_loss + entropy_loss)
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._policy_dist.dist_function().variables)

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
        feed_dict.update({
            self._input_action: action,
            self._input_target_v: target_value,
            self._input_entropy: self._entropy
        })
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={
            "target_value": target_value,
            "pi_loss": self._pi_loss,
            "entropy": self._policy_dist.entropy()
        })


class ActorCritic(sampling.TrajectoryBatchUpdate,
          BaseDeepAgent):
    def __init__(self,
                 f_create_net, state_shape,
                 # ACUpdate arguments
                 discount_factor, entropy=1e-3, target_estimator=None, max_advantage=10.0,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # sampler arguments
                 sampler=None,
                 batch_size=32,
                 *args, **kwargs):
        """
        :param f_create_net: function: f_create_net(inputs) => {"pi": dist_pi, "q": q_values},
                in which {inputs} is [input_state],
                {dist_pi} is probability distribution of policy with shape [None, num_actions],
                {q_values} is Q values with shape [None, num_actions];
                or f_create_net(inputs) => {"mean": mean, "stddev": stddev, "v": v},
                in which {mean} {stddev} is mean and stddev if normal distribution for continuous actions,
                {v} is state value.
        :param state_shape:
        :param discount_factor:
        :param entropy: entropy regulator weight.
        :param target_estimator: optional, default to target_estimate.NStepTD
        :type target_estimator.TargetEstimator
        :param max_advantage: advantage regulation: max advantage value in policy gradient step
        :param network_optimizer: optional, default to network.LocalNetworkOptimizer
        :type network_optimizer: network.NetworkOptimizer
        :param max_gradient: optional, max_gradient clip value
        :param sampler: optional, default to sampling.TrajectoryOnSampler.
                if None, a TrajectoryOnSampler will be created using batch_size.
        :type sampler: sampling.Sampler
        :param batch_size: optional, batch_size when creating sampler
        :param args:
        :param kwargs:
        """
        kwargs.update({
            "f_create_net": f_create_net,
            "state_shape": state_shape,
            "discount_factor": discount_factor,
            "entropy": entropy,
            "target_estimator": target_estimator,
            "max_advantage": max_advantage,
            "max_gradient": max_gradient,
            "batch_size": batch_size,
        })
        print "network_optimizer:", network_optimizer
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        if sampler is None:
            sampler = sampling.TrajectoryOnSampler(interval=batch_size)
            kwargs.update({"sampler": sampler})

        super(ActorCritic, self).__init__(*args, **kwargs)
        pi = self.network["pi"]
        # tf.stop_gradient(pi.op)
        if pi is not None:
            # discrete action: pi is categorical probability distribution
            self._pi_function = network.NetworkFunction(self.network["pi"])
            self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")

            self._pi_distribution = distribution.DiscreteDistribution(self._pi_function, self._input_action)
            q = self.network["q"]
            if q is not None:
                # network outputs q
                self._q_function = network.NetworkFunction(q)
                self._v_function = GreedyStateValueFunction(self._q_function)
            else:
                # network output v
                self._v_function = network.NetworkFunction(self.network["v"])
        else:
            # continuous action: mean / stddev represents normal distribution
            dim_action = self.network["mean"].op.shape.as_list()[-1]
            self._input_action = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name="input_action")
            self._pi_function = network.NetworkFunction(
                outputs={"mean": self.network["mean"], "stddev": self.network["stddev"]},
                inputs=self.network.inputs
            )
            self._pi_distribution = distribution.NormalDistribution(self._pi_function, self._input_action)
            self._v_function = network.NetworkFunction(self.network["v"])
            # continuous action: mean / stddev for normal distribution
        if target_estimator is None:
            # target_estimator = target_estimate.NStepTD(self._v_function, discount_factor)
            target_estimator = target_estimate.GAENStep(self._v_function, discount_factor)
        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(
            ActorCriticUpdater(policy_dist=self._pi_distribution,
                               v_function=self._v_function,
                               target_estimator=target_estimator, entropy=entropy), name="ac")
        # network_optimizer.add_updater(network.L2(self.network), name="l2")
        network_optimizer.compile()

        self._policy = StochasticPolicy(self._pi_distribution)
        # self._policy = GreedyStochasticPolicy(self._pi_distribution)


    def init_network(self, f_create_net, state_shape, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        return network.Network([input_state], f_create_net, var_scope="learn")

    def update_on_trajectory(self, batch):
        self.network_optimizer.update("ac", self.sess, batch)
        # self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        return info, {}

    def set_session(self, sess):
        super(ActorCritic, self).set_session(sess)
        self.network.set_session(sess)
        self._pi_distribution.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)

