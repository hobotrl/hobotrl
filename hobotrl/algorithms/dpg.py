# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl
import hobotrl.network as network
import hobotrl.sampling as sampling
import hobotrl.target_estimate as target_estimate
import hobotrl.tf_dependent.distribution as distribution
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.policy import OUExplorationPolicy, OUNoise
import hobotrl.async as async


class DPGUpdater(network.NetworkUpdater):
    def __init__(self, actor, critic, target_estimator, discount_factor, actor_weight):
        """

        :param actor:
        :type actor network.NetworkFunction
        :param critic:
        :type critic network.NetworkFunction
        :param target_estimator:
        :type target_estimator: target_estimate.TargetEstimator
        """
        super(DPGUpdater, self).__init__()
        self._actor, self._critic, self._target_estimator = \
            actor, critic, target_estimator
        self._dim_action = actor.output().op.shape.as_list()[-1]
        op_q = critic.output().op
        with tf.name_scope("DPGUpdater"):
            with tf.name_scope("input"):
                self._input_target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_q")
                self._input_action_gradient = tf.placeholder(dtype=tf.float32,
                                                             shape=[None, self._dim_action],
                                                             name="input_action_gradient")
            with tf.name_scope("critic"):
                self._critic_loss = tf.reduce_mean(network.Utils.clipped_square(
                    self._input_target_q - op_q
                ))
            with tf.name_scope("actor"):
                # critic.inputs[1] is input_action
                self._action_gradient = tf.gradients(critic.output().op, critic.inputs[1])[0]
                self._gradient_func = network.NetworkFunction(
                    outputs=network.NetworkSymbol(self._action_gradient, "gradient", critic.network),
                    inputs=critic.inputs
                )
                self._actor_loss = tf.reduce_sum(actor.output().op * self._input_action_gradient, axis=1)
                self._actor_loss = -tf.reduce_mean(self._actor_loss)

            self._op_loss = self._actor_loss * actor_weight + self._critic_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._actor.variables +
                                                               self._critic.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done = batch["state"], \
                                                          batch["action"], \
                                                          batch["reward"], \
                                                          batch["next_state"], \
                                                          batch["episode_done"]
        target_q = self._target_estimator.estimate(**batch)
        current_action = self._actor(state)
        action_gradient = self._gradient_func(state, current_action)
        feed_dict = self._critic.input_dict(state, action)
        feed_dict.update(self._actor.input_dict(state))
        feed_dict.update({
            self._input_target_q: target_q,
            self._input_action_gradient: action_gradient
        })

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={
            "target_value": target_q,
            "actor_loss": self._actor_loss,
            "critic_loss": self._critic_loss,
            "loss": self._op_loss,
            "output_action": self._actor.output().op,
            "action_gradient": self._action_gradient
        })


class DPG(sampling.TransitionBatchUpdate,
          BaseDeepAgent):
    def __init__(self,
                 f_se, f_actor, f_critic,
                 state_shape, dim_action,
                 # ACUpdate arguments
                 discount_factor, target_estimator=None,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # policy arguments
                 ou_params=(0.0, 0.2, 0.2),
                 # target network sync arguments
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 # sampler arguments
                 sampler=None,
                 batch_size=32,
                 update_interval=4,
                 replay_size=1000,
                 noise_type=OUNoise,
                 *args, **kwargs):
        """

        :param f_create_net: function, f_create_net([state, action]) => {"q": op_q, "action": op_action}
        :param state_shape: state shape
        :param dim_action: action dimension
        :param discount_factor:
        :param target_estimator: default to target_estimate.ContinuousActionEstimator
        :type target_estimator: target_estimate.TargetEstimator
        :param network_optimizer: default to network.LocalOptimizer
        :type network_optimizer: network.NetworkOptimizer
        :param max_gradient:
        :param ou_params: (mu, theta, sigma) of OU noise arguments
        :param target_sync_interval:
        :param target_sync_rate:
        :param sampler: default to sampling.TransitionSampler
        :type sampler: sampling.Sampler
        :param batch_size:
        :param args:
        :param kwargs:
        """
        kwargs.update({
            "f_se": f_se,
            "f_actor": f_actor,
            "f_critic": f_critic,
            "state_shape": state_shape,
            "dim_action": dim_action,
            "discount_factor": discount_factor,
            "target_estimator": target_estimator,
            "max_gradient": max_gradient,
            "batch_size": batch_size,
            "ou_params": ou_params,
        })
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        if sampler is None:
            sampler = sampling.TransitionSampler(hrl.playback.MapPlayback(replay_size), batch_size, update_interval)
        kwargs.update({"sampler": sampler})
        super(DPG, self).__init__(*args, **kwargs)
        self.network_optimizer = network_optimizer
        self._q_function = network.NetworkFunction(self.network["q"])
        self._actor_function = network.NetworkFunction(self.network["action"], inputs=[self.network.inputs[0]])
        net_target = self.network.target
        self._target_q_function = network.NetworkFunction(net_target["q"])
        self._target_actor_function = network.NetworkFunction(net_target["action"], inputs=[self.network.inputs[0]])
        self._target_v_function = network.NetworkFunction(net_target["v"], inputs=[self.network.inputs[0]])
        self._discount_factor = discount_factor
        self.init_updaters_()

        self._policy = OUExplorationPolicy(self._actor_function, *ou_params, noise_type=noise_type)
        self._target_sync_interval = target_sync_interval
        self._target_sync_rate = target_sync_rate
        self._update_count = 0

    def init_network(self, f_se, f_actor, f_critic, state_shape, dim_action, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        input_action = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name="input_action")

        def f(inputs):
            state, action = input_state, input_action
            net_se = network.Network([state], f_se, var_scope="se")
            se = net_se["se"].op
            net_actor = network.Network([se], f_actor, var_scope="actor")
            net_critic = network.Network([se, input_action], f_critic, var_scope="critic")
            net_critic_for_a = net_critic([se, net_actor["action"].op], name_scope="v_critic")
            return {
                "action": net_actor["action"].op,
                "q": net_critic["q"].op,
                "v": net_critic_for_a["q"].op,
            }, {
                "se": net_se,
                "actor": net_actor,
                "critic": net_critic
            }

        return network.NetworkWithTarget([input_state, input_action],
                                         network_creator=f,
                                         var_scope="learn",
                                         target_var_scope="target")

    def init_updaters_(self):
        target_estimator = target_estimate.ContinuousActionEstimator(
            self._target_v_function, self._discount_factor)
        self.network_optimizer.add_updater(
            DPGUpdater(actor=self._actor_function,
                       critic=self._q_function,
                       target_estimator=target_estimator,
                       discount_factor=self._discount_factor, actor_weight=0.1), name="ac")
        self.network_optimizer.add_updater(network.L2(self.network), name="l2")
        self.network_optimizer.compile()

    def update_on_transition(self, batch):
        self._update_count += 1
        self.network_optimizer.update("ac", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {}

    def set_session(self, sess):
        super(DPG, self).set_session(sess)
        self.network.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)

