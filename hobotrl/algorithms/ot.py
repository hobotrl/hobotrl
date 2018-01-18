#
# -*- coding: utf-8 -*-
# LEARNING TO PLAY IN A DAY: FASTER DEEP REINFORCEMENT LEARNING BY OPTIMALITY TIGHTENING
# https://arxiv.org/abs/1611.01606


import tensorflow as tf
import numpy as np

from dqn import DQN
from hobotrl.sampling import TruncateTrajectorySampler
from hobotrl.playback import MapPlayback
from hobotrl.target_estimate import OptimalityTighteningEstimator
from value_based import GreedyStateValueFunction, DoubleQValueFunction
import hobotrl.network as network


class TrajectoryFitQ(network.FitTargetQ):
    def __init__(self, learn_q, target_estimator):
        """
        :param learn_q: Q function to update
        :type learn_q: network.NetworkFunction
        :param target_estimator: TargetEstimator, default to OptimalityTighteningEstimator
        :type target_estimator: OptimalityTighteningEstimator
        """
        super(TrajectoryFitQ, self).__init__(learn_q, target_estimator)

    def update(self, sess, batch, *args, **kwargs):
        all_state, all_action, all_target = [], [], []
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
        all_state = np.concatenate(all_state)
        all_action = np.concatenate(all_action)
        all_target = np.concatenate(all_target)
        feed_dict = {self._input_target_q: all_target, self._input_action: all_action}
        feed_dict.update(self._q.input_dict(all_state))
        if "_weight" in batch:
            feed_dict[self._input_sample_weight] = batch["_weight"]
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"target_q": all_target,
                                                                  "td_loss": self._sym_loss,
                                                                  "td_losses": self._op_losses})


class OTDQN(DQN):
    def __init__(self, f_create_q,
                 lower_weight, upper_weight, neighbour_size,
                 state_shape, num_actions, discount_factor, ddqn,
                 target_sync_interval, target_sync_rate,
                 greedy_epsilon,
                 network_optimizer=None, max_gradient=10.0,
                 update_interval=4,
                 replay_size=1000, batch_size=32, sampler=None,
                 *args, **kwargs):
        if sampler is None:
            sampler = TruncateTrajectorySampler(MapPlayback(replay_size), batch_size, neighbour_size, update_interval)
        self._lower_weight, self._upper_weight = lower_weight, upper_weight
        super(OTDQN, self).__init__(f_create_q, state_shape, num_actions, discount_factor, ddqn,
                                    target_sync_interval, target_sync_rate, greedy_epsilon,
                                    network_optimizer, max_gradient, update_interval, replay_size,
                                    batch_size, sampler, *args, **kwargs)

    def init_updaters_(self):
        if self._ddqn:
            self.target_v = DoubleQValueFunction(self.learn_q, self.target_q)
        else:
            self.target_v = GreedyStateValueFunction(self.target_q)
        target_esitmator = OptimalityTighteningEstimator(self.target_v, self._upper_weight, self._lower_weight,
                                                         discount_factor=self._discount_factor)

        self.network_optimizer.add_updater(TrajectoryFitQ(self.learn_q, target_esitmator), name="ot")
        self.network_optimizer.add_updater(network.L2(self.network), name="l2")
        self.network_optimizer.compile()

    def update_on_transition(self, batch):
        self._update_count += 1
        self.network_optimizer.update("ot", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {"score": info["TrajectoryFitQ/ot/td_losses"]}
