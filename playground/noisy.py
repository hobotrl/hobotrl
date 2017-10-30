# -*- coding: utf-8 -*-


import sys
import logging

import numpy as np
import tensorflow as tf

from hobotrl.utils import CappedLinear
from hobotrl.sampling import *
from hobotrl.network import *
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.playback import MapPlayback
from hobotrl.target_estimate import GAENStep
from hobotrl.policy import OUNoise


def norm_vector_scale(v, scale=1.0):
    return v / tf.norm(v, axis=-1, keep_dims=True) * scale


class TransitionPolicyGradientUpdater(NetworkUpdater):

    def __init__(self, func_goal, func_value, net_se, target_estimator):
        """

        :param func_goal: goal generator function
        :type func_goal: NetworkFunction
        :param func_value: function computing value
        :type func_value: NetworkFunction
        :param net_se: state encoder network
        :type net_se: Network
        :param target_estimator:
        :type target_estimator: GAENStep
        """
        super(TransitionPolicyGradientUpdater, self).__init__()
        self._estimator = target_estimator
        self._func_goal, self._func_value = func_goal, func_value
        state_shape = net_se.inputs[0].shape.as_list()
        op_v = func_value.output().op
        with tf.name_scope("input"):
            self._input_target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_v")
            self._input_next_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="input_next_state")
        with tf.name_scope("value"):
            td = self._input_target_v - op_v
            self._v_loss = tf.reduce_mean(network.Utils.clipped_square(td))
        with tf.name_scope("goal"):
            # compute advantage
            advantage = self._input_target_v - op_v
            self._advantage = tf.stop_gradient(advantage)
            _mean, _var = tf.nn.moments(advantage, axes=[0])
            self._std_advantage = tf.stop_gradient(advantage / (tf.sqrt(_var) + 1.0))

            net_se_next = net_se([self._input_next_state], "next_se")
            goal_fact = tf.stop_gradient(net_se_next["se"].op - net_se["se"].op)
            goal_predict = func_goal.output().op
            self._goal_fact = goal_fact
            # cosine:
            cos = tf.reduce_sum(goal_fact * goal_predict, axis=-1) / (tf.norm(goal_fact, axis=-1)*tf.norm(goal_predict, axis=-1))
            self._manager_loss = tf.reduce_mean(cos * self._std_advantage)
            self._op_loss = - self._manager_loss + self._v_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=func_goal.variables +
                                                               func_value.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done, noise = batch["state"], \
                                                                 batch["action"], \
                                                                 batch["reward"], \
                                                                 batch["next_state"], \
                                                                 batch["episode_done"], \
                                                                 batch["noise"]

        target_value = self._estimator.estimate(state, action, reward, next_state, episode_done)
        feed_dict = self._func_value.input_dict(state)
        feed_dict.update(self._func_goal.input_dict(state, noise))
        feed_dict.update({
            self._input_next_state: next_state,
            self._input_target_v: target_value
        })
        fetch_dict = {
            "advantage": self._advantage,
            "std_advantage": self._std_advantage,
            "target_value": target_value,
            "v_loss": self._v_loss,
            "manager_loss": self._manager_loss,
            "goal_predict": self._func_goal.output().op,
            "goal_fact": self._goal_fact,
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)


class DisentangleUpdater(NetworkUpdater):
    def __init__(self, net_se, func, stddev=1.0):
        super(DisentangleUpdater, self).__init__()
        self._stddev = stddev
        state_shape = net_se.inputs[0].shape.as_list()
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
            stddev_loss = Utils.clipped_square(stddev - self._input_stddev)
            self._op_loss = mean_loss + stddev_loss
            self._mean_op, self._stddev_op, self._mean_loss, self._stddev_loss = \
                mean, stddev, mean_loss, stddev_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=func.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        all_state, all_noise = [], []
        for trajectory in batch:
            state, noise = trajectory["state"], trajectory["noise"]
            all_state.append(state)
            all_noise.append(noise)
        all_state = np.concatenate(all_state)
        all_noise = np.concatenate(all_noise)
        feed_dict = {
            self._input_state: all_state,
            self._input_noise: all_noise,
            self._input_stddev: self._stddev
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"mean": self._mean_op,
                                                                  "stddev": self._stddev_op,
                                                                  "mean_loss": self._mean_loss,
                                                                  "stddev_loss": self._stddev_loss})
    

class IKUpdater(NetworkUpdater):

    def __init__(self, net_se, net_ik):
        super(IKUpdater, self).__init__()
        state_shape = net_se.inputs[0].shape.as_list()
        action_shape = net_ik["action"].op.shape.as_list()
        with tf.name_scope("input"):
            self._input_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St")
            self._input_next_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St1")
            self._input_action = tf.placeholder(dtype=tf.float32, shape=action_shape, name="action")
        with tf.name_scope("ik"):
            net_se_off = net_se([self._input_state], "off_se")
            net_next_se_off = net_se([self._input_next_state], "off_next_se")
            goal = net_next_se_off["se"].op - net_se_off["se"].op
            goal = norm_vector_scale(goal)
            net_ik_off = net_ik([net_se_off["se"].op, goal], "off_ik")
            action_off = net_ik_off["action"].op
            action_loss = tf.reduce_mean(
                tf.reduce_sum(
                    Utils.clipped_square(self._input_action - action_off),
                    axis=-1)
            )
            self._goal_op, self._action_off_op, self._action_loss = goal, action_off, action_loss
        self._update_operation = network.MinimizeLoss(self._action_loss,
                                                      var_list=net_se.variables+net_ik.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        # horizon c = 1
        all_state, all_next_state, all_action = [], [], []
        for trajectory in batch:
            state, next_state, action = trajectory["state"], trajectory["next_state"], trajectory["action"]
            all_state.append(state)
            all_next_state.append(next_state)
            all_action.append(action)
        all_state = np.concatenate(all_state)
        all_next_state = np.concatenate(all_next_state)
        all_action = np.concatenate(all_action)
        feed_dict = {
            self._input_state: all_state,
            self._input_next_state: all_next_state,
            self._input_action: all_action
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"goal": self._goal_op,
                                                                  "action_off": self._action_off_op,
                                                                  "abs_all_action": np.abs(all_action),
                                                                  "action_loss": self._action_loss,
                                                                  "action": all_action})


class NoisySD(BaseDeepAgent):
    def __init__(self, f_se,    # state encoder             se = SE(s)
                 f_manager,     # manager outputs Dstate    sd = Manager(se)
                 f_explorer,    # noisy network             n  = Noise(z), goal = sd + n
                 f_ik,          # inverse kinetic           a  = IK(se, goal)
                 f_value,       # value network             v  = V(se)
                 f_model,       # transition model          goal[0] = TM(se[0], a[0])
                 state_shape,
                 action_dimension,
                 noise_dimension,
                 discount_factor,
                 target_estimator=None,
                 max_advantage=10.0,
                 network_optimizer=None,
                 max_gradient=10.0,
                 noise_stddev=1.0,
                 noise_explore_param=(0, 0.2, 0.2),
                 worker_explore_param=(0, 0.2, 0.2),
                 manager_horizon=32,
                 batch_size=32,
                 batch_horizon=4,
                 replay_size=1000,
                 *args, **kwargs):
        kwargs.update({
            "f_se": f_se,  # state encoder
            "f_manager": f_manager,  # manager outputs Dstate
            "f_explorer": f_explorer,  # noisy network
            "f_ik": f_ik,  # inverse kinetic
            "f_value": f_value,  # value network
            "f_model": f_model,  # transition model
            "state_shape": state_shape,
            "action_dimension": action_dimension,
            "noise_dimension": noise_dimension,
            "discount_factor": discount_factor,
            "target_estimator": target_estimator,
            "max_advantage": max_advantage,
            "network_optimizer": network_optimizer,
            "max_gradient": max_gradient,
            "batch_size": batch_size,
            "replay_size": replay_size,
            "manager_horizon": manager_horizon,
            "batch_horizon": batch_horizon,
            "noise_stddev": noise_stddev,
            "noise_explore_param": noise_explore_param,
            "worker_explore_param": worker_explore_param
        })
        super(NoisySD, self).__init__(*args, **kwargs)

        def make_sample(state, action, reward, next_state, episode_done, noise, **kwargs):
            sample = default_make_sample(state, action, reward, next_state, episode_done)
            sample.update({"noise": noise})
            return sample

        self._on_data = TrajectoryOnSampler(MapPlayback(capacity=manager_horizon),
                                            interval=manager_horizon, sample_maker=make_sample)
        self._off_data = TruncateTrajectorySampler(MapPlayback(capacity=replay_size),
                                                   batch_size=batch_size,
                                                   trajectory_length=batch_horizon,
                                                   interval=sys.maxint,
                                                   sample_maker=make_sample)

        if network_optimizer is None:
            network_optimizer = LocalOptimizer(tf.contrib.opt.NadamOptimizer(1e-4), max_gradient)
        self._optimizer = network_optimizer
        func_goal = NetworkFunction(self.network["goal"])
        func_value = NetworkFunction(self.network["value"], [self._input_state])
        self._func_action = NetworkFunction(self.network["action"])
        self._func_action_with_goals = NetworkFunction({"action": self.network["action"],
                                                        "sd": self.network["sd"],
                                                        "norm_sd": self.network["norm_sd"],
                                                        "goal": self.network["goal"],
                                                        "noise": self.network["noise"]})
        if target_estimator is None:
            target_estimator = GAENStep(func_value, discount_factor)
        self._optimizer.add_updater(TransitionPolicyGradientUpdater(func_goal,
                                                                    func_value,
                                                                    self.network.sub_net("se"),
                                                                    target_estimator), name="tpg")
        self._optimizer.add_updater(DisentangleUpdater(
            self.network.sub_net("se"),
            self.network.sub_net("noise"),
            stddev=noise_stddev,
        ), name="disentangle")
        self._optimizer.add_updater(IKUpdater(self.network.sub_net("se"), self.network.sub_net("ik")), name="ik")
        self._optimizer.compile()

        # noise generator source for manager targets
        self._noise_source = OUNoise([noise_dimension], *noise_explore_param)
        self._last_input_noise = None
        # noise source for worker actions
        self._noise_action = OUNoise([action_dimension], *worker_explore_param)


    def init_network(self, f_se, f_manager, f_explorer, f_ik, f_value, f_model,
                     state_shape, action_dimension, noise_dimension, discount_factor, *args, **kwargs):
        with tf.name_scope("inputs"):
            self._input_state = tf.placeholder(dtype=np.float32, shape=[None]+list(state_shape), name="input_state")
            self._input_noise = tf.placeholder(dtype=np.float32, shape=[None, noise_dimension], name="input_noise")
            self._input_action = tf.placeholder(dtype=np.float32, shape=[None, action_dimension], name="input_action")

        def f_noisy(inputs):
            # prediction pass
            input_state, input_noise, input_action = inputs[0], inputs[1], inputs[2]
            state_encoder_net = Network([input_state], f_se, "state_encoder")
            encoded_state = state_encoder_net["se"].op
            manager_net = Network([encoded_state], f_manager, "manager")
            sd = manager_net["sd"].op
            norm_sd = norm_vector_scale(sd)
            noise_generator_net = Network([encoded_state, self._input_noise], f_explorer, "noise")
            sd_noise = noise_generator_net["noise"].op
            goal = norm_sd + sd_noise
            goal = norm_vector_scale(goal)
            ik_net = Network([encoded_state, goal], f_ik, "ik")
            value_net = Network([encoded_state], f_value, "value")
            model_net = Network([encoded_state, input_action], f_model, "model")
            return {
                       "action": ik_net["action"].op,
                       "goal": goal, "sd": sd, "noise": sd_noise, "norm_sd": norm_sd,
                       "value": value_net["value"].op
                   }, \
                   {
                       "se": state_encoder_net,
                       "manager": manager_net,
                       "noise": noise_generator_net,
                       "ik": ik_net,
                       "value": value_net,
                       "model": model_net
                   }
            # on: manager, value, state encoder noise
            # off: noise, IK, model

        return Network([self._input_state, self._input_noise, self._input_action], f_noisy, "noisy_policy")

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        super(NoisySD, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        on_batch = self._on_data.step(state, action, reward, next_state, episode_done, noise=self._last_input_noise)
        if on_batch is not None:
            off_batch = self._off_data.step(state, action, reward, next_state, episode_done,
                                            force_sample=True, noise=self._last_input_noise)
        else:
            off_batch = None

        if on_batch is None or off_batch is None:
            return {}
        # on / off batch are synced
        self._optimizer.update("tpg", self.sess, on_batch)
        self._optimizer.update("disentangle", self.sess, off_batch)
        self._optimizer.update("ik", self.sess, off_batch)
        info = self._optimizer.optimize_step(self.sess)
        return info

    def act(self, state, **kwargs):
        n = self._noise_source.tick()
        self._last_input_noise = n
        # action = self._func_action(state[np.newaxis, :], n[np.newaxis, :])[0]
        result = self._func_action_with_goals(state[np.newaxis, :], n[np.newaxis, :])
        action, sd, norm_sd, goal, noise = result["action"][0], result["sd"][0], result["norm_sd"][0], result["goal"][0], result["noise"][0]
        logging.warning("action:%s, sd:%s, norm_sd:%s, goal:%s, noise:%s", action, sd, norm_sd, goal, noise)
        action_exp = self._noise_action.tick()
        action = action + action_exp
        action = np.clip(action, -1.0, 1.0)
        logging.warning("actual action:%s, action_noise:%s", action, action_exp)
        return action




