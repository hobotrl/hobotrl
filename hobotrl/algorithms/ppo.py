# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
import numpy as np


from hobotrl.playback import BatchIterator, MapPlayback, to_columnwise, to_rowwise
import hobotrl.network as network
import hobotrl.sampling as sampling
import hobotrl.target_estimate as target_estimate
import hobotrl.tf_dependent.distribution as distribution
from hobotrl.tf_dependent.base import BaseDeepAgent
from value_based import GreedyStateValueFunction
from hobotrl.policy import StochasticPolicy


class PPOUpdater(network.NetworkUpdater):
    def __init__(self, policy_dist, old_dist, v_function, old_v_function, target_estimator,
                 entropy=1e-1, clip_epsilon=0.1, value_weight=1.0):
        """
        :param policy_dist:
        :type policy_dist: distribution.NNDistribution
        :param old_dist:
        :type old_dist: distribution.NNDistribution
        :param v_function: Function calculating state value
        :type v_function: network.NetworkFunction
        :param old_v_function: Function calculation old state value
        :type old_v_function: network.NetworkFunction
        :param target_estimator:
        :type target_estimator:
        :param entropy: entropy weight, c2 in paper
        :param value_weight: value function loss weight, c1 in paper
        :param clip_epsilon: clipped value of prob ratio
        """
        super(PPOUpdater, self).__init__()
        self._policy_dist, self._old_dist = policy_dist, old_dist
        self._v_function, self._old_v_function = v_function, old_v_function
        self._target_estimator = target_estimator
        self._entropy = entropy
        with tf.name_scope("PPOUpdater"):
            with tf.name_scope("input"):
                self._input_target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_v")
                self._input_action = policy_dist.input_sample()
                self._input_entropy = tf.placeholder(dtype=tf.float32, shape=[], name="input_entropy")
            op_v = v_function.output().op
            old_op_v = old_v_function.output().op
            with tf.name_scope("value"):
                td = self._input_target_v - op_v
                org_v_loss = network.Utils.clipped_square(td)
                clipped_v = old_op_v + tf.clip_by_value(op_v - old_op_v, -clip_epsilon, clip_epsilon)
                clip_v_loss = network.Utils.clipped_square(self._input_target_v - clipped_v)
                self._v_loss = tf.reduce_mean(tf.maximum(org_v_loss, clip_v_loss))
                self._org_v_loss, self._clip_v_loss = org_v_loss, clip_v_loss
            with tf.name_scope("policy"):
                advantage = self._input_target_v - op_v
                self._advantage = advantage
                _mean, _var = tf.nn.moments(advantage, axes=[0])
                self._std_advantage = tf.stop_gradient(advantage / (tf.sqrt(_var) + 1.0))
                ratio = tf.exp(policy_dist.log_prob() - old_dist.log_prob())
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                pi_loss = tf.reduce_mean(tf.minimum(ratio * self._std_advantage, clipped_ratio * self._std_advantage))
                entropy_loss = tf.reduce_mean(self._policy_dist.entropy())
                self._pi_loss = pi_loss
                self._ratio, self._clipped_ratio = ratio, clipped_ratio
            self._op_loss = value_weight * self._v_loss - (self._pi_loss + self._input_entropy * entropy_loss)
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
            "target_value": target_value,
            "pi_loss": self._pi_loss,
            "pi_ratio": self._ratio,
            "pi_ratio_clip": self._clipped_ratio,
            "v_loss": self._v_loss,
            "v_loss_org": self._org_v_loss,
            "v_loss_clip": self._clip_v_loss,
            "entropy": self._policy_dist.entropy(),
            "log_prob": self._policy_dist.log_prob(),
            "advantage_std": self._std_advantage,
        }
        if isinstance(self._policy_dist, distribution.NormalDistribution):
            fetch_dict.update({
                "stddev": self._policy_dist.stddev(),
                "mean": self._policy_dist.mean()
            })
        else:
            pass
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)


class PPO(sampling.TrajectoryBatchUpdate,
          BaseDeepAgent):
    def __init__(self,
                 f_create_net, state_shape,
                 # PPO arguments
                 discount_factor, entropy=1e-3, clip_epsilon=0.2,
                 # update arguments
                 epoch_per_step=4,
                 # target estimate
                 target_estimator=None,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # sampler arguments
                 sampler=None,
                 batch_size=32,
                 horizon=1024,

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
            "max_gradient": max_gradient,
            "batch_size": batch_size,
            "horizon": horizon,
            "clip_epsilon": clip_epsilon,
            "epoch_per_step": epoch_per_step,
        })
        print "network_optimizer:", network_optimizer
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        if sampler is None:
            sampler = sampling.TrajectoryOnSampler(interval=horizon)
            kwargs.update({"sampler": sampler})

        super(PPO, self).__init__(*args, **kwargs)

        self._epoch_py_step = epoch_per_step
        self._batch_size = batch_size

        pi = self.network["pi"]
        if pi is not None:
            # discrete action: pi is categorical probability distribution
            self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
            self._pi_function = network.NetworkFunction(self.network["pi"])
            self._pi_distribution = distribution.DiscreteDistribution(self._pi_function, self._input_action)
            self._old_pi_function = network.NetworkFunction(self._old_network["pi"])
            self._old_pi_distribution = distribution.DiscreteDistribution(self._old_pi_function, self._input_action)
            q = self.network["q"]
            if q is not None:
                # network outputs q
                self._q_function = network.NetworkFunction(q)
                self._v_function = GreedyStateValueFunction(self._q_function)
                self._old_q_function = network.NetworkFunction(self._old_network["q"])
                self._old_v_function = GreedyStateValueFunction(self._old_q_function)
            else:
                # network output v
                self._v_function = network.NetworkFunction(self.network["v"])
                self._old_v_function = network.NetworkFunction(self._old_network["v"])
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
            self._old_pi_function = network.NetworkFunction(
                outputs={"mean": self._old_network["mean"], "stddev": self._old_network["stddev"]},
                inputs=self._old_network.inputs
            )
            self._old_pi_distribution = distribution.NormalDistribution(self._old_pi_function, self._input_action)
            self._old_v_function = network.NetworkFunction(self._old_network["v"])
        if target_estimator is None:
            target_estimator = target_estimate.GAENStep(self._v_function, discount_factor)
        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(
            PPOUpdater(policy_dist=self._pi_distribution,
                       old_dist=self._old_pi_distribution,
                       v_function=self._v_function,
                       old_v_function=self._old_v_function,
                       target_estimator=target_estimator,
                       entropy=entropy,
                       clip_epsilon=clip_epsilon), name="ppo")
        network_optimizer.add_updater(network.L2(self.network), name="l2")
        network_optimizer.compile()

        self._policy = StochasticPolicy(self._pi_distribution)

    def init_network(self, f_create_net, state_shape, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        net = network.Network([input_state], f_create_net, var_scope="learn")
        self._old_network = network.Network([input_state], f_create_net, var_scope="old")
        self._old_network_syncer = network.NetworkSyncer(net, self._old_network)
        return net

    def update_on_trajectory(self, batch):
        # here we receive batch of size horizon.
        infos = []
        info = {}
        for i in range(self._epoch_py_step):
            for mini_batch in BatchIterator(batch, self._batch_size):
                self.network_optimizer.update("ppo", self.sess, mini_batch)
                self.network_optimizer.update("l2", self.sess)
                info = self.network_optimizer.optimize_step(self.sess)
                infos.append(info)
        self._old_network_syncer.sync(self.sess, 1.0)
        return to_columnwise(infos), {}

    def set_session(self, sess):
        super(PPO, self).set_session(sess)
        self.network.set_session(sess)
        self._old_network.set_session(sess)
        self._pi_distribution.set_session(sess)
        self._old_pi_distribution.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)

