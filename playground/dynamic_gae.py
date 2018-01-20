import logging

import numpy as np
from playground import dpg_ot

from hobotrl import DQN, network, playback, DPG, target_estimate
from hobotrl.algorithms.dpg import DPGUpdater
from hobotrl.algorithms.ot import TrajectoryFitQ
from hobotrl.algorithms.value_based import DoubleQValueFunction, GreedyStateValueFunction
from hobotrl.playback import MapPlayback
from hobotrl.sampling import TruncateTrajectorySampler, default_make_sample
from hobotrl.target_estimate import TargetEstimator


class DynamicGAENStep(TargetEstimator):
    """
    target value, based on generalized advantage estimator
    https://arxiv.org/abs/1506.02438
    """

    def __init__(self, v_function, discount_factor=0.99, lambda_decay=0.95, generation_decay=0.95):
        super(DynamicGAENStep, self).__init__(discount_factor)
        self._v, self._lambda_decay = v_function, lambda_decay
        self._generation_decay = generation_decay

    def estimate(self, state, action, reward, next_state, episode_done, generation, current_gen, **kwargs):
        batch_size = len(state)

        states = [s for s in state]
        if not episode_done[-1]:
            states.append(next_state[-1])  # need last next_state
        state_values = self._v(np.asarray(states))
        if episode_done[-1]:
            state_values = np.append(state_values, 0.0)
        # 1 step TD
        delta = state_values[1:] * self._discount_factor + reward - state_values[:-1]

        # approximate generation decay
        generation_decay = self._generation_decay ** (current_gen - np.min(generation))
        logging.warning("generation_decay:%s", generation_decay)
        factor = (generation_decay * self._lambda_decay * self._discount_factor) ** np.arange(batch_size)
        advantage = [np.sum(factor * delta)] \
                    + [np.sum(factor[:-i] * delta[i:]) for i in range(1, batch_size)]

        # accurate generation decay
        # index = np.arange(batch_size)
        # factor = (generation_decay[0] * self._lambda_decay * self._discount_factor) ** index
        # advantage = [np.sum(factor * delta)] \
        #             + [np.sum(((generation_decay[i] * self._lambda_decay * self._discount_factor) ** index[:-i])
        #                       * delta[i:]) for i in range(1, batch_size)]
        target_value = np.asarray(advantage) + state_values[:-1]
        return target_value


class TruncateTrajectorySamplerWithLatest(TruncateTrajectorySampler):
    """Sample {batch_size} trajectories of length {trajectory_length} in every
    {interval} steps.
    Make sure the nearest added trajectory is sampled
    """
    def __init__(self, replay_memory=None, batch_size=8, trajectory_length=8, interval=4, sample_maker=None):
        super(TruncateTrajectorySamplerWithLatest, self).__init__(replay_memory, batch_size, trajectory_length, interval, sample_maker)

    def step(self, state, action, reward, next_state, episode_done, force_sample=False, **kwargs):
        """
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :param force_sample: boolean, True if sample batch immediately ignoring interval setting.
        :param kwargs:
        :return: list of dict, each dict is a column-wise batch of transitions in a trajectory
        """
        trajectories = super(TruncateTrajectorySamplerWithLatest, self).step(state, action, reward, next_state, episode_done, force_sample, **kwargs)
        if trajectories is None:
            return None
        nearest_index = (self._replay.push_index - 1 + self._replay.get_capacity()) % self._replay.get_capacity()
        nearest_sample = playback.to_rowwise(self._replay.get_batch([nearest_index]))[0]
        nearest_trajectory = self._trajectory_near(nearest_sample)
        if nearest_trajectory is not None:
            # substitute with nearest
            trajectories[-1] = nearest_trajectory
        return trajectories


class OnDQN(DQN):
    def __init__(self, f_create_q,
                 neighbour_size,
                 state_shape, num_actions, discount_factor, ddqn,
                 target_sync_interval, target_sync_rate,
                 greedy_epsilon,
                 network_optimizer=None, max_gradient=10.0,
                 update_interval=4,
                 replay_size=1000, batch_size=32, sampler=None,
                 generation_decay=0.95,
                 *args, **kwargs):

        if sampler is None:
            def make(state, action, reward, next_state, episode_done, **kwargs):
                sample = default_make_sample(state, action, reward, next_state, episode_done)
                sample.update({"generation": self._update_count})
                return sample
            sampler = TruncateTrajectorySamplerWithLatest(MapPlayback(replay_size), batch_size, neighbour_size, update_interval,
                                                sample_maker=make)
        self._generation_decay = generation_decay
        super(OnDQN, self).__init__(f_create_q, state_shape, num_actions, discount_factor, ddqn,
                                    target_sync_interval, target_sync_rate, greedy_epsilon,
                                    network_optimizer, max_gradient, update_interval, replay_size,
                                    batch_size, sampler, *args, **kwargs)

    def init_updaters_(self):
        if self._ddqn:
            self.target_v = DoubleQValueFunction(self.learn_q, self.target_q)
        else:
            self.target_v = GreedyStateValueFunction(self.target_q)
        target_esitmator = DynamicGAENStep(self.target_v, discount_factor=self._discount_factor,
                                           generation_decay=self._generation_decay)

        self.network_optimizer.add_updater(TrajectoryFitQ(self.learn_q, target_esitmator), name="ot")
        self.network_optimizer.add_updater(network.L2(self.network), name="l2")
        self.network_optimizer.compile()

    def update_on_transition(self, batch):
        for traj in batch:
            traj["current_gen"] = self._update_count
        self._update_count += 1
        self.network_optimizer.update("ot", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {"score": info["TrajectoryFitQ/ot/td_losses"]}


class OnDPG(DPG):
    def __init__(self, f_se, f_actor, f_critic, state_shape, dim_action, discount_factor, target_estimator=None,
                 network_optimizer=None, max_gradient=10.0, ou_params=(0.0, 0.2, 0.2), target_sync_interval=10,
                 target_sync_rate=0.01, sampler=None, batch_size=8, update_interval=4, replay_size=1000,
                 generation_decay=0.95, neighbour_size=8,
                 *args,
                 **kwargs):

        if sampler is None:
            def make(state, action, reward, next_state, episode_done, **kwargs):
                sample = default_make_sample(state, action, reward, next_state, episode_done)
                sample.update({"generation": self._update_count})
                return sample
            sampler = TruncateTrajectorySamplerWithLatest(MapPlayback(replay_size), batch_size, neighbour_size, update_interval,
                                                sample_maker=make)
        self._generation_decay = generation_decay
        self._neighbour_size = neighbour_size
        super(OnDPG, self).__init__(f_se, f_actor, f_critic, state_shape, dim_action, discount_factor, target_estimator,
                                    network_optimizer, max_gradient, ou_params, target_sync_interval, target_sync_rate,
                                    sampler, batch_size, update_interval, replay_size, *args, **kwargs)

    def init_updaters_(self):
        target_estimator = DynamicGAENStep(
            self._target_v_function, self._discount_factor, generation_decay=self._generation_decay)
        self.network_optimizer.add_updater(
            dpg_ot.TrajectoryFitQ(actor=self._actor_function,
                       critic=self._q_function,
                       target_estimator=target_estimator,
                       discount_factor=self._discount_factor, actor_weight=0.1), name="ac")
        self.network_optimizer.add_updater(network.L2(self.network), name="l2")
        self.network_optimizer.compile()

    def update_on_transition(self, batch):
        for traj in batch:
            traj["current_gen"] = self._update_count
        self._update_count += 1
        self.network_optimizer.update("ac", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {}

