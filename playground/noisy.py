# -*- coding: utf-8 -*-


import sys
import logging

import numpy as np
import tensorflow as tf

from hobotrl.sampling import *
from hobotrl.network import *
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.playback import MapPlayback
from hobotrl.target_estimate import GAENStep
from hobotrl.policy import OUNoise
from hobotrl.policy import OUExplorationPolicy
from hobotrl.tf_dependent.distribution import NormalDistribution
from hobotrl.algorithms import ac as ac


def norm_vector_scale(v, scale=1.0):
    return v / tf.norm(v, axis=-1, keep_dims=True) * scale


def cosine(a, b, name=None):
    cos = tf.div(tf.reduce_sum(a * b, axis=-1), (tf.norm(a, axis=-1) * tf.norm(b, axis=-1)), name=name)
    return cos


class TransitionPolicyGradientUpdater(NetworkUpdater):

    def __init__(self, func_goal, func_value, net_se, target_estimator, abs_goal=False):
        """

        :param func_goal: goal generator function
        :type func_goal: NetworkFunction
        :param func_value: function computing value
        :type func_value: NetworkFunction
        :param net_se: state encoder network
        :type net_se: Network
        :param target_estimator:
        :type target_estimator: GAENStep
        :param abs_goal: False if goal represents direction, True if goal represents state delta
        :type abs_goal: bool
        """
        super(TransitionPolicyGradientUpdater, self).__init__()
        self._estimator = target_estimator
        self._func_goal, self._func_value = func_goal, func_value
        state_shape = net_se.inputs[0].shape.as_list()
        se_dimension = net_se["se"].op.shape.as_list()[-1]

        op_v = func_value.output().op
        self._op_v = op_v
        with tf.name_scope("input"):
            self._input_target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_v")
            self._input_next_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="input_next_state")
        with tf.name_scope("value"):
            td = self._input_target_v - op_v
            self._v_loss = tf.reduce_mean(network.Utils.clipped_square(td, 2.0))
        with tf.name_scope("goal"):
            # compute advantage
            advantage = self._input_target_v - op_v
            self._advantage = tf.stop_gradient(advantage)
            _mean, _var = tf.nn.moments(advantage, axes=[0])
            self._std_advantage = tf.stop_gradient(advantage / (tf.sqrt(_var) + 1.0))

            net_se_next = net_se([self._input_next_state], "next_se")
            goal_fact = tf.stop_gradient(net_se_next["se"].op - net_se["se"].op)
            goal_predict = func_goal.output().op
            self._goal_predict_norm = tf.norm(goal_predict, axis=-1)
            self._goal_fact = goal_fact
            self._goal_fact_norm = tf.norm(goal_fact, axis=-1)
            self._se = net_se["se"].op
            self._next_se = net_se_next["se"].op
            # cosine transition policy loss
            cos = cosine(goal_fact, goal_predict, name="cosine_goal")
            self._manager_cos = cos
            self._manager_norm_delta = Utils.clipped_square(self._goal_fact_norm - self._goal_predict_norm)
            if not abs_goal:
                self._manager_loss = tf.reduce_mean(cos * self._std_advantage)
            else:
                self._manager_loss = tf.reduce_mean((cos - 1e-1 * self._manager_norm_delta) * self._std_advantage)
            self._op_loss = self._v_loss - self._manager_loss
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
            "current_value": self._op_v,
            "loss_v": self._v_loss,
            "loss_manager": self._manager_loss,
            "los_cos": self._manager_cos,
            "goal_predict": self._func_goal.output().op,
            "goal_fact": self._goal_fact,
            "goal_predict_norm": self._goal_predict_norm,
            "goal_fact_norm": self._goal_fact_norm,
            "se": self._se,
            "next_se": self._next_se
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)


class AchievableUpdator(NetworkUpdater):
    def __init__(self, func_goal, net_se):
        super(AchievableUpdator, self).__init__()
        self._func_goal = func_goal
        self._net_state = net_se(name_scope="state")
        self._net_next_state = net_se(name_scope="next_state")
        self._goal_fact = self._net_next_state["se"].op - self._net_state["se"].op
        self._goal_predict = func_goal.output().op
        self._achievable_loss = tf.reduce_mean(Utils.clipped_square(self._goal_fact - self._goal_predict))
        self._update_operation = network.MinimizeLoss(self._achievable_loss,
                                                      var_list=func_goal.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done, noise = batch["state"], \
                                                                 batch["action"], \
                                                                 batch["reward"], \
                                                                 batch["next_state"], \
                                                                 batch["episode_done"], \
                                                                 batch["noise"]
        feed_dict = {
            self._net_state.inputs[0]: state,
            self._net_next_state.inputs[0]: next_state,
        }
        feed_dict.update(self._func_goal.input_dict(state, noise))
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={
            "loss": self._achievable_loss,
            "goal_fact": self._goal_fact,
            "goal_predict": self._goal_predict
        })


class DisentangleUpdater(NetworkUpdater):
    def __init__(self, net_se, func, stddev=1.0):
        super(DisentangleUpdater, self).__init__()
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
            self._op_loss = mean_loss + stddev_loss
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
    

class ModelUpdater(NetworkUpdater):
    def __init__(self, net_se, net_model, reward_weight=1.0):
        super(ModelUpdater, self).__init__()
        state_shape = net_se.inputs[0].shape.as_list()
        se_shape = net_se["se"].op.shape.as_list()
        action_shape = net_model.inputs[1].shape.as_list()
        with tf.name_scope("input"):
            self._input_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St")
            self._input_next_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St1")
            self._input_action = tf.placeholder(dtype=tf.float32, shape=action_shape, name="action")
            self._input_reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        with tf.name_scope("forward_model"):
            se = net_se([self._input_state])["se"].op
            next_se = net_se([self._input_next_state])["se"].op
            self._goal_fact = tf.stop_gradient(next_se - se)
            logging.warning("ModelUpdater model input:%s", [se, self._input_action])
            net = net_model([se, self._input_action])
            self._goal_predict = net["sd"].op
            self._goal_loss = tf.reduce_mean(
                tf.reduce_sum(
                    Utils.clipped_square(self._goal_fact - self._goal_predict)
                ))
            self._reward_loss = tf.reduce_mean(Utils.clipped_square(self._input_reward - net["r"].op))
            self._loss = self._goal_loss + reward_weight * self._reward_loss
        self._update_operation = network.MinimizeLoss(self._loss,
                                                      var_list=net_se.variables + net_model.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        # horizon c = 1
        if isinstance(batch, list):
            batch = to_transitions(batch)
        state, next_state, action, reward = \
            batch["state"], batch["next_state"], batch["action"], batch["reward"]

        feed_dict = {
            self._input_state: state,
            self._input_next_state: next_state,
            self._input_action: action,
            self._input_reward: reward,
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"goal_predict": self._goal_predict,
                                                                  "goal_fact": self._goal_fact,
                                                                  "action": self._input_action,
                                                                  "loss_reward": self._reward_loss,
                                                                  "loss_goal": self._goal_loss,
                                                                  "loss": self._loss})


class ImaginaryGoalGradient(NetworkUpdater):
    """
    update IK with imaginary goals
    """
    def __init__(self, net_se, net_model, net_ik, func_goal=None):
        super(ImaginaryGoalGradient, self).__init__()
        self._func_goal = func_goal
        state_shape = net_se.inputs[0].shape.as_list()
        se_shape = net_se["se"].op.shape.as_list()
        action_shape = net_model["sd"].op.shape.as_list()
        with tf.name_scope("input"):
            self._input_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St")
            self._input_goal_random = tf.placeholder(dtype=tf.float32, shape=se_shape, name="goal_random")
            self._input_goal_history = tf.placeholder(dtype=tf.float32, shape=se_shape, name="goal_history")

        with tf.name_scope("imagine"):
            se = net_se([self._input_state])["se"].op
            norm_scale = tf.stop_gradient(
                tf.reduce_mean(tf.norm(self._input_goal_history, axis=-1))
                / tf.reduce_mean(tf.norm(self._input_goal_random, axis=-1))
            )
            self._goal_fact = tf.stop_gradient(norm_scale) * self._input_goal_random
            action_predict = net_ik([se, self._goal_fact])["action"].op
            self._goal_predict = net_model([se, action_predict])["sd"].op
            self._loss = tf.reduce_mean(
                tf.reduce_sum(
                    Utils.clipped_square(self._goal_fact - self._goal_predict), axis=-1)
            )
        self._update_operation = network.MinimizeLoss(self._loss,
                                                      var_list=net_ik.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        # horizon c = 1
        if isinstance(batch, list):
            batch = to_transitions(batch)
        state, next_state, action, noise, goal = \
            batch["state"], batch["next_state"], batch["action"], batch["noise"], batch["goal"]
        # generates goals
        if self._func_goal is not None:
            goal_random = self._func_goal(state, noise)
        else:
            goal_random = np.random.rand(*goal.shape)
        feed_dict = {
            self._input_state: state,
            self._input_goal_history: goal,
            self._input_goal_random: goal_random
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"goal_predict": self._goal_predict,
                                                                  "goal_fact": self._goal_fact,
                                                                  "loss": self._loss})


class IKUpdater(NetworkUpdater):

    def __init__(self, net_se, net_ik, abs_goal=False):
        super(IKUpdater, self).__init__()
        state_shape = net_se.inputs[0].shape.as_list()
        se_shape = net_se["se"].op.shape.as_list()
        action_shape = net_ik["action"].op.shape.as_list()
        with tf.name_scope("input"):
            self._input_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St")
            self._input_next_state = tf.placeholder(dtype=tf.float32, shape=state_shape, name="St1")
            self._input_action = tf.placeholder(dtype=tf.float32, shape=action_shape, name="action")
            self._input_goal = tf.placeholder(dtype=tf.float32, shape=se_shape, name="se")
        with tf.name_scope("ik"):
            net_se_off = net_se([self._input_state], "off_se")
            net_next_se_off = net_se([self._input_next_state], "off_next_se")
            goal = net_next_se_off["se"].op - net_se_off["se"].op
            if not abs_goal:
                norm_scale = tf.stop_gradient(
                    tf.reduce_mean(tf.norm(self._input_goal, axis=-1))
                    / tf.reduce_mean(tf.norm(goal, axis=-1))
                )
                goal = goal * norm_scale
            self._goal_norm = tf.norm(goal, axis=-1)
            net_ik_off = net_ik([net_se_off["se"].op, goal], "off_ik")
            action_off = net_ik_off["action"].op
            action_loss = tf.reduce_mean(
                tf.reduce_sum(
                    Utils.clipped_square(self._input_action - action_off),
                    axis=-1)
            )
            self._goal_complete_loss = cosine(self._input_goal, goal)
            self._goal_op, self._action_off_op, self._action_loss = goal, action_off, action_loss
        self._update_operation = network.MinimizeLoss(self._action_loss,
                                                      var_list=net_se.variables+net_ik.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        # horizon c = 1
        if isinstance(batch, list):
            batch = to_transitions(batch)
        state, next_state, action, goal = \
            batch["state"], batch["next_state"], batch["action"], batch["goal"]

        feed_dict = {
            self._input_state: state,
            self._input_next_state: next_state,
            self._input_action: action,
            self._input_goal: goal
        }
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"goal": self._goal_op,
                                                                  "goal_norm": self._goal_norm,
                                                                  "action_off": self._action_off_op,
                                                                  "abs_all_action": np.abs(action),
                                                                  "loss_action": self._action_loss,
                                                                  "loss_completion": self._goal_complete_loss,
                                                                  "action": action})


class WorkerActorCritic(ac.ActorCriticUpdater):
    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done, noise, goal = batch["state"], \
                                                                       batch["action"], \
                                                                       batch["reward"], \
                                                                       batch["next_state"], \
                                                                       batch["episode_done"], \
                                                                       batch["noise"], \
                                                                       batch["goal"]
        target_value = self._target_estimator.estimate(**batch)
        feed_dict = self._v_function.input_dict(state)
        feed_dict.update(self._policy_dist.dist_function().input_dict(**batch))
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
        if isinstance(self._policy_dist, NormalDistribution):
            fetch_dict.update({
                "stddev": self._policy_dist.stddev(),
                "mean": self._policy_dist.mean()
            })
        else:
            pass
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)


class NoisySD(BaseDeepAgent):
    def __init__(self, f_se,    # state encoder             se = SE(s)
                 f_manager,     # manager outputs Dstate    sd = Manager(se)
                 f_explorer,    # noisy network             n  = Noise(z), goal = sd + n
                 f_ik,          # inverse kinetic           a  = IK(se, goal)
                 f_value,       # value network             v  = V(se)
                 f_model,       # transition model          goal[0] = TM(se[0], a[0])
                 f_pi,          # worker_policy
                 state_shape,
                 action_dimension,
                 noise_dimension,
                 se_dimension,
                 discount_factor,
                 target_estimator=None,
                 max_advantage=10.0,
                 network_optimizer=None,
                 max_gradient=10.0,
                 noise_stddev=1.0,
                 noise_explore_param=(0, 0.2, 0.2),
                 worker_explore_param=(0, 0.2, 0.2),
                 worker_entropy=1e-2,
                 manager_horizon=32,
                 manager_interval=2,
                 manager_entropy=1e-2,
                 batch_size=32,
                 batch_horizon=4,
                 replay_size=1000,
                 act_ac=False,          # True if act by actor critic; False by IK
                 intrinsic_weight=0.0,
                 explore_net=False,     # True if explore by noise net; False by plain ou noise
                 abs_goal=True,         # True if goal expressed in absolute delta; False in direction
                 manager_ac=False,      # True if manager trained with actor critic
                 achievable_weight=1e-1,
                 disentangle_weight=1.0,
                 *args, **kwargs):
        kwargs.update({
            "f_se": f_se,  # state encoder
            "f_manager": f_manager,  # manager outputs Dstate
            "f_explorer": f_explorer,  # noisy network
            "f_ik": f_ik,  # inverse kinetic
            "f_value": f_value,  # value network
            "f_model": f_model,  # transition model
            "f_pi": f_pi,  # actor_critic
            "state_shape": state_shape,
            "action_dimension": action_dimension,
            "noise_dimension": noise_dimension,
            "se_dimension": se_dimension,
            "discount_factor": discount_factor,
            "target_estimator": target_estimator,
            "max_advantage": max_advantage,
            "network_optimizer": network_optimizer,
            "max_gradient": max_gradient,
            "batch_size": batch_size,
            "replay_size": replay_size,
            "manager_horizon": manager_horizon,
            "manager_interval": manager_interval,
            "manager_entropy": manager_entropy,
            "batch_horizon": batch_horizon,
            "noise_stddev": noise_stddev,
            "noise_explore_param": noise_explore_param,
            "worker_explore_param": worker_explore_param,
            "worker_entropy": worker_entropy,
            "disentangle_weight": disentangle_weight,
        })

        # algorithm hyperparameter
        # True if act by actor critic; False by IK
        self._act_ac = act_ac
        self._intrinsic_weight = intrinsic_weight
        # True if explore by noise net; False by plain ou noise
        self._explore_net = explore_net
        # True if goal expressed in absolute delta; False in direction
        self._abs_goal = abs_goal
        # True if manager trained with actor critic
        self._manager_ac = manager_ac

        super(NoisySD, self).__init__(*args, **kwargs)

        def make_sample(state, action, reward, next_state, episode_done, noise, goal, goal_step, **kwargs):
            sample = default_make_sample(state, action, reward, next_state, episode_done)
            sample.update({"noise": noise, "goal": goal, "goal_step": goal_step})
            return sample

        self._on_data = TrajectoryOnSampler(interval=manager_horizon, sample_maker=make_sample)
        # self._off_data = TruncateTrajectorySampler(MapPlayback(capacity=replay_size),
        #                                            batch_size=batch_size,
        #                                            trajectory_length=batch_horizon,
        #                                            interval=sys.maxint,
        #                                            sample_maker=make_sample)
        self._off_data = TransitionSampler(MapPlayback(capacity=replay_size), batch_size=batch_size,
                                           interval=sys.maxint, sample_maker=make_sample)
        if network_optimizer is None:
            network_optimizer = LocalOptimizer(tf.contrib.opt.NadamOptimizer(1e-4), max_gradient)
        self._optimizer = network_optimizer
        func_goal = NetworkFunction(self.network["goal"], inputs=[self._input_state, self._input_noise])
        func_value = NetworkFunction(self.network["value"], [self._input_state])
        self._func_manager = func_goal
        self._func_se = NetworkFunction(self.network.sub_net("se")["se"])
        goal_op = self.network["goal"].op
        self._func_worker = NetworkFunction(self.network["action"], inputs=[self._input_state, goal_op])
        self._func_worker_ac = NetworkFunction(outputs={
            "mean": self.network["worker_mean"],
            "stddev": self.network["worker_stddev"],
        }, inputs=[self._input_state, goal_op])

        if target_estimator is None:
            target_estimator = GAENStep(func_value, discount_factor)

        if not self._manager_ac:
            self._optimizer.add_updater(TransitionPolicyGradientUpdater(func_goal,
                                                                        func_value,
                                                                        self.network.sub_net("se"),
                                                                        target_estimator,
                                                                        abs_goal=self._abs_goal),
                                        name="tpg", forward_pass="manager")
        else:
            self._manager_pi_function = network.NetworkFunction(
                outputs={"mean": self.network["goal"], "stddev": self.network["manager_stddev"]},
                inputs={"state": self._input_state}
            )
            input_goal = tf.placeholder(dtype=tf.float32, shape=[None, se_dimension], name="input_goal")
            self._manager_pi_dist = NormalDistribution(self._manager_pi_function, input_goal)
            self._optimizer.add_updater(ac.ActorCriticUpdater(self._manager_pi_dist,
                                                              func_value,
                                                              target_estimator,
                                                              manager_entropy)
                                        , name="manager_ac", forward_pass="manager")
        self._optimizer.add_updater(AchievableUpdator(func_goal, self.network.sub_net("se")),
                                    weight=achievable_weight, name="achievable", forward_pass="manager")
        if self._explore_net:
            self._optimizer.add_updater(DisentangleUpdater(
                self.network.sub_net("se"),
                self.network.sub_net("noise"),
                stddev=noise_stddev,
            ), weight=disentangle_weight, name="disentangle")
        if not self._act_ac:
            self._optimizer.add_updater(IKUpdater(
                self.network.sub_net("se"),
                self.network.sub_net("ik")), name="ik")
            self._optimizer.add_updater(ModelUpdater(
                self.network.sub_net("se"),
                self.network.sub_net("model")), name="model")
            self._optimizer.add_updater(ImaginaryGoalGradient(
                self.network.sub_net("se"),
                self.network.sub_net("model"),
                self.network.sub_net("ik"), func_goal=None), name="imagine")
        else:
            # worker  pi
            self._pi_function = network.NetworkFunction(
                outputs={"mean": self.network["worker_mean"], "stddev": self.network["worker_stddev"]},
                inputs={"state": self._input_state, "goal": goal_op}
            )
            self._pi_dist = NormalDistribution(self._pi_function, self._input_action)
            # self._worker_v_function = NetworkFunction(self.network["worker_value"], inputs=[self._input_state, self._input_noise])
            # todo worker's state should be the concatenation of encoded_state and goal.

            func_worker_value = NetworkFunction(self.network["worker_value"], inputs=[self._input_state])
            worker_estimator = GAENStep(func_worker_value, discount_factor)
            self._func_ac_act = NetworkFunction(outputs={
                "mean": self.network["worker_mean"],
                "stddev": self.network["worker_stddev"],
                "goal": self.network["goal"],
            })
            self._optimizer.add_updater(WorkerActorCritic(self._pi_dist, func_worker_value, worker_estimator, worker_entropy), name="worker_ac")
        self._optimizer.add_updater(L2(self.network), name="l2")
        self._optimizer.compile()

        # noise generator source for manager targets
        self._noise_source = OUNoise([noise_dimension], *noise_explore_param)
        self._manager_interval = manager_interval
        self._last_input_noise = None
        self._last_goal = None
        self._last_goal_step = 0
        # noise source for worker actions
        self._noise_action = OUNoise([action_dimension], *worker_explore_param)

    def init_network(self, f_se, f_manager, f_explorer, f_ik, f_value, f_model, f_pi,
                     state_shape, action_dimension, noise_dimension, se_dimension, discount_factor, *args, **kwargs):
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
            noise_generator_net = Network([tf.stop_gradient(encoded_state), self._input_noise], f_explorer, "noise")
            sd_noise = noise_generator_net["noise"].op
            if self._explore_net:
                goal = sd + sd_noise
            else:
                goal = sd
            # goal = norm_vector_scale(goal)
            ik_net = Network([encoded_state, goal], f_ik, "ik")
            value_net = Network([encoded_state], f_value, "value")

            # actor critic worker
            worker_state = tf.concat((encoded_state, tf.stop_gradient(goal)), axis=-1, name="worker_state")
            worker_value_net = Network([encoded_state], f_value, "worker_value")
            worker_pi_net = Network([worker_state], f_pi, "worker_pi")
            logging.warning("model input:%s", [encoded_state, input_action])
            model_net = Network([encoded_state, input_action], f_model, "model")
            return {
                       "action": ik_net["action"].op,
                       "goal": goal, "sd": sd, "noise": sd_noise, "norm_sd": norm_sd,
                       "manager_stddev": manager_net["stddev"].op,
                       "value": value_net["value"].op,
                       "worker_value": worker_value_net["value"].op,
                       "worker_mean": worker_pi_net["mean"].op,
                       "worker_stddev": worker_pi_net["stddev"].op,
                       "worker_state": worker_state,
                   }, \
                   {
                       "se": state_encoder_net,
                       "manager": manager_net,
                       "noise": noise_generator_net,
                       "ik": ik_net,
                       "value": value_net,
                       "worker_value": worker_value_net,
                       "worker_pi": worker_pi_net,
                       "model": model_net,
                   }
            # on: manager, value, state encoder noise
            # off: noise, IK, model

        return Network([self._input_state, self._input_noise, self._input_action], f_noisy, "noisy_policy")

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        super(NoisySD, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        on_batch = self._on_data.step(state, action, reward, next_state, episode_done,
                                      noise=self._last_input_noise, goal=self._last_goal, goal_step=self._last_goal_step)
        if on_batch is not None:
            off_batch = self._off_data.step(state, action, reward, next_state, episode_done,
                                            force_sample=True,
                                            noise=self._last_input_noise, goal=self._last_goal, goal_step=self._last_goal_step)
        else:
            off_batch = None
        if episode_done:
            # need to regenerate goal from manager
            self._last_goal = None

        if on_batch is None or off_batch is None:
            return {}
        # on / off batch are synced
        # pick samples according to goal_step
        rows = MapPlayback.to_rowwise(on_batch)
        manager_batch = []
        manager_sample = None
        for sample in rows:
            if sample["goal_step"] == 1:
                if manager_sample is not None:
                    manager_sample["next_state"] = sample["state"]
                    manager_batch.append(manager_sample)
                manager_sample = sample
            else:
                if manager_sample is not None:
                    manager_sample["reward"] += sample["reward"]
                    manager_sample["next_state"] = sample["next_state"]
                    if sample["episode_done"]:
                        manager_sample["episode_done"] = sample["episode_done"]
                        manager_batch.append(manager_sample)
                        manager_sample = None
        if manager_sample is not None:
            manager_batch.append(manager_sample)
        manager_batch = MapPlayback.to_columnwise(manager_batch)
        if self._abs_goal:
            # absolute goal saved in batch; relative goal need for training
            manager_batch["goal"] = manager_batch["goal"] - self._func_se(manager_batch["state"])
            off_batch["goal"] = off_batch["goal"] - self._func_se(off_batch["state"])
        manager_batch["action"] = manager_batch["goal"]
        if self._manager_ac:
            self._optimizer.update("manager_ac", self.sess, manager_batch)
        else:
            self._optimizer.update("tpg", self.sess, manager_batch)
        self._optimizer.update("achievable", self.sess, manager_batch)
        if self._explore_net:
            self._optimizer.update("disentangle", self.sess, off_batch)
        if not self._act_ac:
            # self._optimizer.update("ik", self.sess, on_batch)
            self._optimizer.update("ik", self.sess, off_batch)
            self._optimizer.update("model", self.sess, off_batch)
            self._optimizer.update("imagine", self.sess, off_batch)
        else:
            # recalculate reward for worker
            goal_truth = self._func_se(on_batch["next_state"]) - self._func_se(on_batch["state"])
            goal = on_batch["goal"]
            org_reward = on_batch["reward"]
            reward = np.sum(goal * goal_truth, axis=-1)
            norm1 = np.linalg.norm(goal, axis=-1)
            norm2 = np.linalg.norm(goal_truth, axis=-1)
            reward = reward / (norm1 * norm2)
            on_batch["reward"] = self._intrinsic_weight * reward + org_reward
            self._optimizer.update("worker_ac", self.sess, on_batch)
        self._optimizer.update("l2", self.sess)
        info = self._optimizer.optimize_step(self.sess)
        return info

    def act(self, state, **kwargs):
        if not self._act_ac:
            return self.act_ik(state, **kwargs)
        else:
            return self.act_ac(state, **kwargs)

    def check_goal(self, state, **kwargs):
        exploration = False
        goal_shape = self._func_manager.output().op.shape.as_list()[1:]
        if self._last_goal is None or self._last_goal_step >= self._manager_interval:
            # need to regenerate goal
            if self._explore_net:
                # explore via generative network
                n = self._noise_source.tick()
            else:
                n = np.zeros(self._noise_source._shape, dtype=np.float32)
            if self._manager_ac:
                result = self._manager_pi_function(state[np.newaxis, :])
                mean, stddev = result["mean"], result["stddev"]
                goal = self._manager_pi_dist.do_sample(mean, stddev)[0]
            else:
                if not self._explore_net and np.random.rand() < 0.2:
                    exploration = True
                    goal = np.random.normal(0.0, 1.0, goal_shape)
                else:
                    exploration = False
                    goal = self._func_manager(state[np.newaxis, :], n[np.newaxis, :])[0]
            if self._abs_goal:
                # save absolute target goal
                se = self._func_se(state[np.newaxis, :])[0]
                self._last_goal = se + goal
            else:
                # save directional goal
                self._last_goal = goal
            self._last_input_noise = n
            self._last_goal_step = 0
        self._last_goal_step += 1

    def get_relative_goal(self, state):
        state_array = np.asarray(state)[np.newaxis, :]
        if self._abs_goal:
            se = self._func_se(state_array)[0]
            goal = self._last_goal - se
        else:
            goal = self._last_goal
        return goal

    def act_ac(self, state, **kwargs):
        self.check_goal(state, **kwargs)
        goal = self.get_relative_goal(state)
        result = self._func_worker_ac(np.asarray(state)[np.newaxis, :], goal[np.newaxis, :])
        mean, stddev = result["mean"], result["stddev"]
        logging.warning("step:%s, goal:%s, last_goal:%s, mean:%s, stddev:%s, n:%s", self._last_goal_step, goal, self._last_goal, mean, stddev, self._last_input_noise)
        action = self._pi_dist.do_sample(mean, stddev)[0]
        logging.warning("action:%s", action)
        return action

    def act_ik(self, state, **kwargs):
        self.check_goal(state, **kwargs)
        goal = self.get_relative_goal(state)
        action = self._func_worker(state[np.newaxis, :], goal[np.newaxis, :])[0]
        action_exp = self._noise_action.tick()
        action_all = OUExplorationPolicy.action_add(action, action_exp)
        logging.warning("step:%s, goal:%s, last_goal:%s, action_ik:%s, action_noise:%s", self._last_goal_step, goal, self._last_goal, action, action_all)
        return action_all
