#
# -*- coding: utf-8 -*-
# LEARNING TO PLAY IN A DAY: FASTER DEEP REINFORCEMENT LEARNING BY OPTIMALITY TIGHTENING
# https://arxiv.org/abs/1611.01606


import tensorflow as tf
import numpy as np

from hobotrl.algorithms.dpg import DPG, DPGUpdater
from hobotrl.sampling import TruncateTrajectorySampler
from hobotrl.playback import MapPlayback
from hobotrl.target_estimate import OptimalityTighteningEstimator
from hobotrl.algorithms.value_based import GreedyStateValueFunction
import hobotrl.network as network


class TrajectoryFitQ(DPGUpdater):
    def __init__(self, actor, critic, target_estimator, discount_factor, actor_weight):
        super(TrajectoryFitQ, self).__init__(actor, critic, target_estimator, discount_factor, actor_weight)

    def update(self, sess, batch, *args, **kwargs):
        all_state, all_action, all_target, all_action_gradient = [], [], [], []
        for trajectory in batch:
            state, action, reward, next_state, episode_done = trajectory["state"], \
                                                              trajectory["action"], \
                                                              trajectory["reward"], \
                                                              trajectory["next_state"], \
                                                              trajectory["episode_done"]
            all_state.append(state)
            all_action.append(action)
            target = self._target_estimator.estimate(state, action, reward, next_state, episode_done)
            all_target.append(target)
            current_action = self._actor(state)
            action_gradient = self._gradient_func(state, current_action)
            all_action_gradient.append(action_gradient)
        all_state = np.concatenate(all_state)
        all_action = np.concatenate(all_action)
        all_target = np.concatenate(all_target)
        all_action_gradient = np.concatenate(all_action_gradient)
        feed_dict = self._critic.input_dict(all_state, all_action)
        feed_dict.update(self._actor.input_dict(all_state))
        feed_dict.update({
            self._input_target_q: all_target,
            self._input_action_gradient: all_action_gradient,
        })

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={
            "target_value": all_target,
            "actor_loss": self._actor_loss,
            "critic_loss": self._critic_loss,
            "loss": self._op_loss,
            "output_action": self._actor.output().op,
            "action_gradient": self._action_gradient,
        })


class OTDPG(DPG):

    def __init__(self, f_se, f_actor, f_critic,
                 lower_weight, upper_weight, neighbour_size,
                 state_shape, dim_action, discount_factor, target_estimator=None,
                 network_optimizer=None, max_gradient=10.0, ou_params=(0.0, 0.2, 0.2), target_sync_interval=10,
                 target_sync_rate=0.01, sampler=None, batch_size=32, update_interval=4, replay_size=1000, *args,
                 **kwargs):
        if sampler is None:
            sampler = TruncateTrajectorySampler(MapPlayback(replay_size), batch_size, neighbour_size, 4)
        self._lower_weight, self._upper_weight, self._neighbour_size = lower_weight, upper_weight, neighbour_size
        super(OTDPG, self).__init__(f_se, f_actor, f_critic, state_shape, dim_action, discount_factor, target_estimator,
                                    network_optimizer, max_gradient, ou_params, target_sync_interval, target_sync_rate,
                                    sampler, batch_size, update_interval, replay_size, *args, **kwargs)

    def init_updaters_(self):
        target_esitmator = OptimalityTighteningEstimator(self._target_v_function, self._upper_weight, self._lower_weight,
                                                         discount_factor=self._discount_factor)

        self.network_optimizer.add_updater(TrajectoryFitQ(self._actor_function,
                                                          self._q_function,
                                                          target_esitmator,
                                                          self._discount_factor, 0.1), name="ac")
        self.network_optimizer.add_updater(network.L2(self.network), name="l2")
        self.network_optimizer.compile()
