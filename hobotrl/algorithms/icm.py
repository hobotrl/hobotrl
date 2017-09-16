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
            self._op_loss = 200 * self._forward_loss

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
        feed_dict = self._forward_function.input_dict(state, next_state, action)
        # feed_dict.update(self._forward_function.input_dict(action))
        # feed_dict.update(self._feature_function.input_dict(next_state))
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"forward loss": self._forward_loss})


class InverseUpdater(network.NetworkUpdater):
    def __init__(self, inverse_function, policy_dist):
        super(InverseUpdater, self).__init__()
        self._inverse_function, self._policy_dist = inverse_function, policy_dist

        with tf.name_scope("InverseUpdater"):
            with tf.name_scope("input"):
                self._input_action = policy_dist.input_sample()

            op_action_hat = inverse_function.output().op

            # inverse loss calculation
            with tf.name_scope("inverse"):
                inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(indices=self._input_action, depth=np.shape(op_action_hat)[1], on_value=1,
                                      off_value=0, axis=-1),
                    logits=op_action_hat)
                )
                self._inverse_loss = inverse_loss

            self._op_loss = self._inverse_loss

        self._update_operation = network.MinimizeLoss(self._op_loss, var_list=self._inverse_function.variables +
                                                      self._policy_dist._dist_function.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done = batch["state"], \
                                                          batch["action"], \
                                                          batch["reward"], \
                                                          batch["next_state"], \
                                                          batch["episode_done"]

        feed_dict = self._policy_dist.dist_function().input_dict(state)
        feed_dict.update(self._inverse_function.input_dict(state, next_state, action))
        feed_more = {self._input_action: action}
        feed_dict.update(feed_more)

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"inverse loss": self._inverse_loss})


class ActorCriticWithICM(sampling.TrajectoryBatchUpdate,
          BaseDeepAgent):
    def __init__(self,
                 f_se, f_ac, f_forward, f_inverse, state_shape,
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

        def f_icm(inputs):
            f_se1 = network.Network([inputs[0]], f_se, var_scope='learn_se1')
            f_se1 = network.NetworkFunction(f_se1["se"]).output().op
            f_se2 = network.Network([inputs[1]], f_se, var_scope='learn_se2')
            f_se2 = network.NetworkFunction(f_se2["se"]).output().op

            f_ac_out = network.Network([f_se1], f_ac, var_scope='learn_ac')
            v = network.NetworkFunction(f_ac_out["v"]).output().op
            pi_dist = network.NetworkFunction(f_ac_out["pi"]).output().op

            one_hot_action = tf.one_hot(indices=inputs[2], depth=2, on_value=1, off_value=0, axis=-1)
            one_hot_action = tf.cast(one_hot_action, tf.float32)
            f_forward_out = network.Network([one_hot_action, f_se1], f_forward, var_scope='learn_forward')
            phi2_hat = network.NetworkFunction(f_forward_out["phi2_hat"]).output().op

            f_inverse_out = network.Network([f_se1, f_se2], f_inverse, var_scope='learn_inverse')
            logits = network.NetworkFunction(f_inverse_out["logits"]).output().op

            bonus = 200 * 0.5 * tf.reduce_sum(tf.square(f_se2 - phi2_hat), axis=1)

            return {"pi": pi_dist, "v": v, "logits": logits, "phi1": f_se1, "phi2": f_se2, "phi2_hat": phi2_hat,
                    "bonus": bonus}

        kwargs.update({
            "f_icm": f_icm,
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

        super(ActorCriticWithICM, self).__init__(*args, **kwargs)

        pi = self.network["pi"]

        if pi is not None:
            # discrete action: pi is categorical probability distribution
            self._pi_function = network.NetworkFunction(self.network["pi"])
            # self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")

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
        self._phi2_hat_function = network.NetworkFunction(self.network["phi2_hat"])
        self._phi2_function = network.NetworkFunction(self.network["phi2"])
        self._logits = network.NetworkFunction(self.network["logits"])
        self._bonus = network.NetworkFunction(self.network["bonus"])

        if target_estimator is None:
            target_estimator = target_estimate.NStepTD(self._v_function, discount_factor, bonus=self._bonus)
            # target_estimator = target_estimate.GAENStep(self._v_function, discount_factor)
        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(
            ActorCriticUpdater(policy_dist=self._pi_distribution,
                               v_function=self._v_function,
                               target_estimator=target_estimator, entropy=entropy), name="ac"
        )
        network_optimizer.add_updater(network.L2(self.network), name="l2")
        network_optimizer.add_updater(
            ForwardUpdater(forward_function=self._phi2_hat_function,
                           feature_function=self._phi2_function,
                           policy_dist=self._pi_distribution), name="forward"
        )
        network_optimizer.add_updater(
            InverseUpdater(inverse_function=self._logits,
                           policy_dist=self._pi_distribution), name="inverse"
        )
        network_optimizer.compile()

        self._policy = StochasticPolicy(self._pi_distribution)

    def init_network(self, f_icm, state_shape, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        input_next_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape),
                                          name="input_next_state")
        self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
        return network.Network([input_state, input_next_state, self._input_action], f_icm, var_scope="learn")

    def update_on_trajectory(self, batch):
        # self.network_optimizer.update("ac", self.sess, batch)
        # self.network_optimizer.update("l2", self.sess)
        self.network_optimizer.update("forward", self.sess, batch)
        # self.bonus = self.network_optimizer.optimize_step(self.sess)
        # print "--------------bonus---------------", self.bonus
        self.network_optimizer.update("inverse", self.sess, batch)
        self.network_optimizer.update("ac", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        # print "----------------info-------------", info
        return info, {}

    def set_session(self, sess):
        super(ActorCriticWithICM, self).set_session(sess)
        self.network.set_session(sess)
        self._pi_distribution.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)