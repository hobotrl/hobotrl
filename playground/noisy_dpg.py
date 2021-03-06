# -*- coding: utf-8 -*-

import sys

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl
import hobotrl.network as network
import hobotrl.sampling as sampling
import hobotrl.target_estimate as target_estimate
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.policy import OUNoise
from hobotrl.algorithms import dpg
from noisy import DisentangleUpdater


class CenterDisentangleUpdater(network.NetworkUpdater):
    def __init__(self, net_se, func, stddev=1.0, stddev_weight=1e-3):
        super(CenterDisentangleUpdater, self).__init__()
        self._stddev = stddev
        state_shape = net_se.inputs[0].shape.as_list()
        se_dimension = net_se["se"].op.shape.as_list()[-1]
        noise_shape = func.inputs[1].shape.as_list()
        with tf.name_scope("input"):
            self._input_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St")
            self._input_noise = tf.placeholder(dtype=tf.float32, shape=noise_shape, name="Nt")
            self._input_stddev = tf.placeholder(dtype=tf.float32, name="stddev")
        with tf.name_scope("disentangle"):
            net_se_off = net_se([self._input_state], "off_se")
            net_noise_off = func([tf.stop_gradient(net_se_off["se"].op), self._input_noise], "off_noise")
            self._noise_op = net_noise_off["noise"].op

            mean = tf.reduce_mean(self._noise_op, axis=0, keep_dims=True)
            mean_loss = tf.reduce_sum(Utils.clipped_square(mean))
            stddev = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self._noise_op - mean), axis=-1)))
            stddev_loss = Utils.clipped_square(stddev - self._input_stddev * np.sqrt(se_dimension))
            self._op_loss = mean_loss + stddev_loss * stddev_weight
            self._mean_op, self._stddev_op, self._mean_loss, self._stddev_loss = \
                mean, stddev, mean_loss, stddev_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=func.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        if isinstance(batch, list):
            batch = to_transitions(batch)
        state, noise = batch["state"], batch["noise"]
        feed_dict = {
            self._input_state: state,
            self._input_noise: noise,
            self._input_stddev: self._stddev
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"mean": self._mean_op,
                                                                  "stddev": self._stddev_op,
                                                                  "mean_loss": self._mean_loss,
                                                                  "stddev_loss": self._stddev_loss})


class NoisyDPGUpdater(dpg.DPGUpdater):
    def __init__(self, actor, critic, target_estimator, discount_factor, actor_weight, actor_mean):
        """

        :param actor:
        :type actor network.NetworkFunction
        :param critic:
        :type critic network.NetworkFunction
        :param target_estimator:
        :type target_estimator: target_estimate.TargetEstimator
        """
        super(NoisyDPGUpdater, self).__init__(actor, critic, target_estimator, discount_factor, actor_weight)
        self._actor_mean = actor_mean
        with tf.name_scope("NoisyDPGUpdater"):
            with tf.name_scope("action_mean"):
                self._input_action_mean_gradient = tf.placeholder(dtype=tf.float32,
                                                             shape=[None, self._dim_action],
                                                             name="input_action_mean_gradient")
                self._actor_mean_loss = tf.reduce_sum(actor_mean.output().op * self._input_action_mean_gradient, axis=1)
                self._actor_mean_loss = -tf.reduce_mean(self._actor_mean_loss)
            self._op_loss = (self._actor_loss + self._actor_mean_loss) * actor_weight + self._critic_loss
            # self._op_loss = self._actor_mean_loss * actor_weight + self._critic_loss

        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._actor.variables +
                                                               self._critic.variables + self._actor_mean.variables)

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done, noise = batch["state"], \
                                                          batch["action"], \
                                                          batch["reward"], \
                                                          batch["next_state"], \
                                                          batch["episode_done"], batch["noise"]
        target_q = self._target_estimator.estimate(**batch)
        current_action = self._actor(state, noise)
        action_gradient = self._gradient_func(state, current_action)
        action_mean = self._actor_mean(state)
        action_mean_gradient = self._gradient_func(state, action_mean)
        feed_dict = self._critic.input_dict(state, action)
        feed_dict.update(self._actor.input_dict(state, noise))
        feed_dict.update({
            self._input_target_q: target_q,
            self._input_action_gradient: action_gradient
        })
        feed_dict.update(self._actor_mean.input_dict(state))
        feed_dict.update({
            self._input_action_mean_gradient: action_mean_gradient
        })
        logging.warning("target_value:%s, action_gradient:%s, action_mean_gradient:%s", np.mean(target_q), np.mean(action_gradient), np.mean(action_mean_gradient))
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={
            "target_value": target_q,
            "actor_loss": self._actor_loss,
            "critic_loss": self._critic_loss,
            "actor_mean_loss": self._actor_mean_loss,
            "loss": self._op_loss,
            "output_action": self._actor.output().op,
            "action_gradient": self._action_gradient,
            "action_gradient_noise": action_gradient,
            "action_gradient_mean": action_mean_gradient,
            "action_mean": action_mean,
            "target_q": target_q
        })


class DisentangleNoisyDPGUpdater(NoisyDPGUpdater):

    def __init__(self, actor, critic, f_noise, target_estimator, discount_factor, actor_weight, actor_mean,
                 zero_mean_weight=1e-2, stddev_weight=1e-4):
        super(DisentangleNoisyDPGUpdater, self).__init__(actor, critic, target_estimator, discount_factor, actor_weight,
                                                         actor_mean)
        self._f_noise = f_noise
        self._zero_mean_weight, self._stddev_weight = zero_mean_weight, stddev_weight
        with tf.name_scope("disentangle"):
            self._input_weight_mean = tf.placeholder(dtype=tf.float32, name="weight_mean")
            self._input_weight_stddev = tf.placeholder(dtype=tf.float32, name="weight_stddev")
            op_a, op_a_mean = self._actor.output().op, self._actor_mean.output().op
            # pull action mean close to noisy action
            self._zero_mean_loss = network.Utils.clipped_square(tf.stop_gradient(op_a) - op_a_mean)
            # push noisy action away from action mean
            self._stddev_loss = - network.Utils.clipped_square(f_noise.output().op)
            self._disentangle_loss = tf.reduce_mean(self._zero_mean_loss) * self._input_weight_mean \
                                     + tf.reduce_mean(self._stddev_loss) * self._input_weight_stddev
        self._op_loss = (self._actor_loss + self._actor_mean_loss) * actor_weight + self._critic_loss \
                        + self._disentangle_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._actor.variables +
                                                               self._critic.variables + self._actor_mean.variables)

    def update(self, sess, batch, *args, **kwargs):
        run = super(DisentangleNoisyDPGUpdater, self).update(sess, batch, *args, **kwargs)
        run._feed_dict.update({
            self._input_weight_mean: self._zero_mean_weight,
            self._input_weight_stddev: self._stddev_weight
        })
        run._fetch_dict.update({
            "disentangle_mean": self._zero_mean_loss,
            "disentangle_stddev": self._stddev_loss,
            "weight_mean": self._zero_mean_weight,
            "weight_stddev": self._stddev_weight
        })
        return run


class NoisyContinuousActionEstimator(target_estimate.TargetEstimator):
    def __init__(self, actor, critic, discount_factor):
        super(NoisyContinuousActionEstimator, self).__init__(discount_factor)
        self._actor, self._critic, = actor, critic

    def estimate(self, state, action, reward, next_state, episode_done, noise, **kwargs):
        target_action = self._actor(next_state, noise)
        target_q = self._critic(next_state, target_action)
        target_q = reward + self._discount_factor * (1.0 - episode_done) * target_q
        return target_q


class NoisyContinuousActionEstimator(target_estimate.TargetEstimator):
    def __init__(self, actor, critic, discount_factor):
        super(NoisyContinuousActionEstimator, self).__init__(discount_factor)
        self._actor, self._critic, = actor, critic

    def estimate(self, state, action, reward, next_state, episode_done, noise, **kwargs):
        target_action = self._actor(next_state, noise)
        target_q = self._critic(next_state, target_action)
        target_q = reward + self._discount_factor * (1.0 - episode_done) * target_q
        return target_q


class NoisyDPG(BaseDeepAgent):
    def __init__(self,
                 f_se,
                 f_actor,
                 f_critic,
                 f_noise,
                 state_shape, dim_action, dim_noise,
                 # ACUpdate arguments
                 discount_factor,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # policy arguments
                 ou_params=(0.0, 0.2, 0.2),
                 noise_stddev=0.5,
                 noise_weight=1.0,
                 noise_mean_weight=1e-2,
                 noise_stddev_weight=1e-4,
                 # target network sync arguments
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 # sampler arguments
                 replay_size=1000,
                 batch_size=32,
                 disentangle_with_dpg=True,
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
            "f_noise": f_noise,
            "state_shape": state_shape,
            "dim_action": dim_action,
            "dim_noise": dim_noise,
            "discount_factor": discount_factor,
            "max_gradient": max_gradient,
            "batch_size": batch_size,
            "replay_size": replay_size,
            "ou_params": ou_params,
            "noise_stddev": noise_stddev,
            "noise_weight": noise_weight
        })
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        super(NoisyDPG, self).__init__(*args, **kwargs)

        self._disentangle_with_dpg = disentangle_with_dpg

        def make_sample(state, action, reward, next_state, episode_done, noise, **kwargs):
            sample = sampling.default_make_sample(state, action, reward, next_state, episode_done)
            sample.update({"noise": noise})
            return sample

        self._sampler = sampling.TransitionSampler(hrl.playback.MapPlayback(replay_size), batch_size, 4, sample_maker=make_sample)

        self._q_function = network.NetworkFunction(self.network["q"],
                                                   inputs=[self._input_state, self._input_action])
        self._actor_function = network.NetworkFunction(self.network["action"],
                                                       inputs=[self._input_state, self._input_noise])
        self._actor_mean_function = network.NetworkFunction(self.network["action_mean"],
                                                            inputs=[self._input_state])
        self._noise_function = network.NetworkFunction(self.network["action_noise"],
                                                       inputs=[self._input_state, self._input_noise])
        self._target_q_function = network.NetworkFunction(self.network.target["q"],
                                                          inputs=[self._input_state, self._input_action])
        self._target_actor_function = network.NetworkFunction(self.network.target["action"],
                                                              inputs=[self._input_state, self._input_noise])
        target_estimator = NoisyContinuousActionEstimator(
            self._target_actor_function, self._target_q_function, discount_factor)
        self.network_optimizer = network_optimizer
        if disentangle_with_dpg:
            network_optimizer.add_updater(
                DisentangleNoisyDPGUpdater(actor=self._actor_function,
                                           critic=self._q_function,
                                           f_noise=self._noise_function,
                                           target_estimator=target_estimator,
                                           discount_factor=discount_factor, actor_weight=0.02,
                                           actor_mean=self._actor_mean_function,
                                           zero_mean_weight=noise_mean_weight,
                                           stddev_weight=noise_stddev_weight), name="ac")
        else:
            network_optimizer.add_updater(
                NoisyDPGUpdater(actor=self._actor_function,
                                critic=self._q_function,
                                target_estimator=target_estimator,
                                discount_factor=discount_factor, actor_weight=0.02,
                                actor_mean=self._actor_mean_function), name="ac")
            network_optimizer.add_updater(
                DisentangleUpdater(
                    self.network.sub_net("se"),
                    self.network.sub_net("noise"),
                    stddev=noise_stddev), weight=noise_weight,
                name="disentangle")
        network_optimizer.add_updater(network.L2(self.network), name="l2")
        network_optimizer.compile()

        self._act_all_function = network.NetworkFunction({
            "action": self.network["action"],
            "mean": self.network["action_mean"],
            "noise": self.network["action_noise"]
        }, inputs=[self._input_state, self._input_noise])

        self._noise_source = OUNoise([dim_noise], *ou_params)
        self._last_input_noise = None
        # self._policy = OUExplorationPolicy(self._actor_function, *ou_params)
        self._target_sync_interval = target_sync_interval
        self._target_sync_rate = target_sync_rate
        self._update_count = 0

    def init_network(self, f_se, f_actor, f_critic, f_noise, state_shape, dim_action, dim_noise, *args, **kwargs):
        self._input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        self._input_action = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name="input_action")
        self._input_noise = tf.placeholder(dtype=tf.float32, shape=[None, dim_noise], name="input_noise")

        def f(inputs):
            state, action, noise = inputs
            se_net = network.Network([state], f_se, var_scope="se")
            se = se_net["se"].op
            q_net = network.Network([se, action], f_critic, var_scope="critic")
            pi_net = network.Network([se], f_actor, var_scope="actor")
            noise_net = network.Network([tf.stop_gradient(se), noise], f_noise, var_scope="noise")
            a_out, n_out = pi_net["action"].op, noise_net["noise"].op
            action_out = a_out + tf.abs(tf.sign(n_out) - a_out) * tf.tanh(n_out)
            return {
                "se": se,
                "action": action_out,
                "action_mean": a_out,
                "action_noise": n_out,
                "q": q_net["q"].op
            }, {
                "se": se_net,
                "actor": pi_net,
                "critic": q_net,
                "noise": noise_net
            }

        return network.NetworkWithTarget([self._input_state, self._input_action, self._input_noise],
                                         network_creator=f,
                                         var_scope="learn",
                                         target_var_scope="target")

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        super(NoisyDPG, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        batch = self._sampler.step(state, action, reward, next_state, episode_done,
                                   noise=self._last_input_noise, **kwargs)
        if batch is not None:
            self._update_count += 1
            self.network_optimizer.update("ac", self.sess, batch)
            if not self._disentangle_with_dpg:
                self.network_optimizer.update("disentangle", self.sess, batch)
            self.network_optimizer.update("l2", self.sess)
            info = self.network_optimizer.optimize_step(self.sess)
            if self._update_count % self._target_sync_interval == 0:
                self.network.sync_target(self.sess, self._target_sync_rate)
            return info
        return {}

    def act(self, state, **kwargs):
        n = self._noise_source.tick()
        self._last_input_noise = n
        logging.warning("state:%s", state)
        result = self._act_all_function(state[np.newaxis, :], n[np.newaxis, :])
        action, action_mean, action_noise = result["action"][0], result["mean"][0], result["noise"][0]
        logging.warning("action:%s; mean:%s, noise: %s", action, action_mean, action_noise)
        return action

