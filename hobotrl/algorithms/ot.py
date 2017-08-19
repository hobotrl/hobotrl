#
# -*- coding: utf-8 -*-
# LEARNING TO PLAY IN A DAY: FASTER DEEP REINFORCEMENT LEARNING BY OPTIMALITY TIGHTENING
# https://arxiv.org/abs/1611.01606


import tensorflow as tf
import numpy as np

import hobotrl as hrl
import hobotrl.tf_dependent.value_function as vf
from hobotrl.utils import Network


class FitQFunction(vf.DeepQFuncActionOut):
    def __init__(self, gamma, f_net_dqn, state_shape, num_actions, training_params, schedule, batch_size,
                 greedy_policy=True, ddqn=False, graph=None, **kwargs):
        super(FitQFunction, self).__init__(gamma, f_net_dqn, state_shape, num_actions, training_params, schedule,
                                           batch_size, greedy_policy, ddqn, graph, **kwargs)

        self.sym_input_target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_q")

        # for computing td loss
        self.sym_input_td_target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="input_td_target_q")

        with tf.name_scope("fit_target"):
            sym_fit_losses = Network.clipped_square(self.sym_input_target_q - self.sym_q_sel)
            sym_td_losses = Network.clipped_square(self.sym_input_td_target_q - self.sym_q_sel)
            # self.sym_fit_loss = self.sym_regularization_loss + \
            self.sym_fit_loss = tf.reduce_mean(self.sym_importance * sym_fit_losses)
            optimizer = training_params[0]
            self.op_fit_target = optimizer.minimize(self.sym_fit_loss)
            self.sym_td_losses = sym_td_losses

    def fit_target_value(self, state, action, target_value, td_target_value, importance=None, sess=None, **kwargs):
        feed_dict = {
            self.sym_state: state,
            self.sym_action: action,
            self.sym_input_target_q: target_value,
            self.sym_input_td_target_q: td_target_value
        }
        if importance is not None:
            feed_dict[self.sym_importance] = importance

        return sess.run([self.op_fit_target, self.sym_fit_loss, self.sym_td_losses], feed_dict)

    def improve_value_(self,
                       state, action, target_value, td_target_value,
                       importance=None,
                       sess=None,
                       **kwargs):
        """Public Interface for Training Value Fcn.
        The Deep Q-Network training procedure: apply `op_train_td`
        and `op_sync_target` with the periodic schedule specified by
        `self.__N_STEP_TD` and `self.__N_STEP_SYNC`。

        Parameters
        ----------
        :param state: a batch of state
        :param action: a batch of action
        :param target_value: target value to fit
        :param td_target_value: target value to compute td loss only; does not contribute to training loss
        :param importance: a batch of importance, or scalar importance value
        :param sess: tf session
        :param kwargs:
        :return:
        """
        self.countdown_td_ -= 1
        self.countdown_sync_ -= 1
        info = {}
        td_loss = 0
        if self.countdown_td_ == 0:
            _, loss, td_losses = self.fit_target_value(state=state, action=action,
                                                       target_value=target_value,
                                                       td_target_value=td_target_value,
                                                       importance=importance,
                                                       sess=sess, **kwargs)
            self.countdown_td_ = self._N_STEP_TD
            info = {"fit_loss": loss, "target_value": target_value, "td_losses": td_losses}

        if self.countdown_sync_ == 0:
            self.apply_op_sync_target_(sess=sess)
            self.countdown_sync_ = self._N_STEP_SYNC

        return info


class OTValueUpdateMixin(hrl.mixin.BaseValueMixin):
    """
    no not work with ReplayBufferMixin.

    """
    def __init__(self, state_shape, num_actions, batch_size, reward_decay, K, weight_upper_bound, weight_lower_bound,
                 replay_capacity=1000,
                 replay_class=hrl.playback.MapPlayback,
                 priority_bias=1.0,
                 importance_weight=1.0,
                 state_offset_scale=(0, 1),
                 **kwargs):
        kwargs.update({
            "state_shape": state_shape,
            "num_actions": num_actions,
            "batch_size": batch_size,
            "reward_decay": reward_decay,

        })
        super(OTValueUpdateMixin, self).__init__(**kwargs)
        self.q_function = FitQFunction(**kwargs)
        self._BATCH_SIZE = kwargs['batch_size']
        self.state_shape, self.num_actions, self.reward_decay, self.batch_size, self.K, \
            self.weight_upper, self.weight_lower = \
            state_shape, num_actions, reward_decay, batch_size, K, weight_upper_bound, weight_lower_bound

        self.step_n = 0
        self.episode_n = 0  # counter for episode
        self.episode_samples = []  # store samples from this episode, for calculating future_reward
        augment_offset, augment_scale = {}, {}
        if list(state_offset_scale) != [0, 1]:
            augment_offset['state'], augment_scale['state'] = state_offset_scale[0], state_offset_scale[1]
            augment_offset['next_state'], augment_scale['next_state'] = state_offset_scale[0], state_offset_scale[1]

        if replay_class == hrl.playback.MapPlayback:
            self.replay = replay_class(capacity=replay_capacity, sample_shapes={
                "state": state_shape,
                "action": [],
                "reward": [],
                "next_state": state_shape,
                "episode_done": [],
                "future_reward": [],
                "episode_n": [],
            }, augment_scale=augment_scale, augment_offset=augment_offset)
        elif replay_class == hrl.playback.NearPrioritizedPlayback:
            self.replay = replay_class(capacity=replay_capacity, sample_shapes={
                "state": state_shape,
                "action": [],
                "reward": [],
                "next_state": state_shape,
                "episode_done": [],
                "future_reward": [],
                "episode_n": [],
            }, augment_scale=augment_scale, augment_offset=augment_offset,
                                       priority_bias=priority_bias,
                                       importance_weight=importance_weight)

    def improve_value_(self, state, action, reward, next_state, episode_done, **kwargs):
        self.step_n += 1
        self.episode_samples.append({"state": state,
                                     "action": np.asarray(action),
                                     "reward": np.asarray(reward, dtype=float),
                                     "next_state": next_state,
                                     "episode_done": np.asarray(episode_done, dtype=float),
                                     "future_reward": np.asarray(0.0),
                                     "episode_n": np.asarray(self.episode_n)})
        if episode_done:
            # calculate future_reward
            R = 0.0
            for i in range(len(self.episode_samples)):
                sample = self.episode_samples[-i-1]
                R = R * self.reward_decay + sample["reward"]
                sample["future_reward"] = R
            for i in range(len(self.episode_samples)):
                self.replay.push_sample(self.episode_samples[i])
            self.episode_samples = []

        info = {}
        if self.replay.get_count() > self.batch_size * self.K:
            batch = self.replay.sample_batch(self.batch_size)
            index = batch["_index"]
            nearby_index = np.asarray([range(i - self.K, i + self.K + 1) for i in index])
            nearby_index = nearby_index % self.replay.get_count()
            # print "nearby_index:", nearby_index
            nearby_batch = [hrl.playback.MapPlayback.to_rowwise(self.replay.get_batch(i)) for i in nearby_index]
            nearby_batch_index = [(filter(lambda x: x[0]["episode_n"] == n, zip(nb, ni)), i)
                                       for n, i, nb, ni in zip(batch["episode_n"], index, nearby_batch, nearby_index)]
            """
            nearby_batch_index = [
                ([
                    ({"state": state, "action": action, ...}, nearby_index_0),
                    ({"state": state, "action": action, ...}, nearby_index_1),
                    ...,
                    ({"state": state, "action": action, ...}, nearby_index_n),
                ], sample_index_0),
                ([
                    ({"state": state, "action": action, ...}, nearby_index_0),
                    ({"state": state, "action": action, ...}, nearby_index_1),
                    ...,
                    ({"state": state, "action": action, ...}, nearby_index_n),
                ], sample_index_1),
                ...
            ]
            """
            # all 'state'
            nearby_states = [[nb_batch['state'] for nb_batch, nb_i in nb_bi] for nb_bi, b_i in nearby_batch_index]
            # last 'next_state'
            nearby_last_next = [[nb_bi[-1][0]['next_state']] for nb_bi, b_i in nearby_batch_index]
            nearby_states = [a + b for a, b in zip(nearby_states, nearby_last_next)]
            state_targets = []
            for n_s in nearby_states:
                t = self.q_function.get_target_v(state=np.asarray(n_s), sess=self.get_session())
                state_targets.append(t)
            # print "nearby_states:", nearby_states
            # print "state_targets:", state_targets
            # print "nearby_batch_index", nearby_batch_index
            lower_bounds, upper_bounds, target_values, targets0 = [], [], [], []
            for _i in range(len(nearby_batch_index)):
                nb_bi, sample_i = nearby_batch_index[_i]
                targets = state_targets[_i]
                nb_indices = [x[1] for x in nb_bi]
                sample_index = nb_indices.index(sample_i)
                sample_target = targets[sample_index+1] * (1.0 - batch["episode_done"][_i]) * self.reward_decay \
                                + batch["reward"][_i]
                low, high = batch["future_reward"][_i], sample_target + 0.1
                # print "sample_i, nb_indices, sample_index:", sample_i, nb_indices, sample_index
                r = 0.0
                for k in range(1, self.K + 1):  # upper bound: k -> [1, K]
                    neighbor_index = sample_index - k
                    if neighbor_index < 0:
                        break
                    nb_batch = nb_bi[neighbor_index][0]
                    # print "r, nb_batch:", r, nb_batch
                    r += (self.reward_decay ** -k) * nb_batch["reward"]
                    new_high = (self.reward_decay ** -k) * targets[neighbor_index] - r
                    if new_high < high:
                        high = new_high
                r = batch["reward"][_i]
                for k in range(1, self.K + 1):
                    neighbor_index = sample_index + k
                    if neighbor_index >= len(nb_bi):
                        break
                    nb_batch = nb_bi[neighbor_index][0]
                    r += (self.reward_decay ** k) * nb_batch["reward"]
                    if not nb_batch["episode_done"]:
                        new_low = (self.reward_decay ** (k + 1)) * targets[neighbor_index+1] + r
                    else:
                        new_low = r
                    if new_low > low:
                        low = new_low
                # low, high = batch["future_reward"][_i], sys.maxint
                # high = sys.mxaxint
                lower_bounds.append(low)
                upper_bounds.append(high)
                w = 1.0 + self.weight_lower + self.weight_upper
                w0, w_lower, w_upper = 1.0 / w, self.weight_lower / w, self.weight_upper / w
                target_value = sample_target
                if low > sample_target > high:
                    target_value = w0 * sample_target + w_lower * low + w_upper * high
                elif low > sample_target:
                    target_value = (w0 * sample_target + w_lower * low) / (w0 + w_lower)
                elif high < sample_target:
                    target_value = (w0 * sample_target + w_upper * high) / (w0 + w_upper)

                target_values.append(target_value)
                targets0.append(sample_target)

            # print "target_values:", target_values
            # print "upper_bounds:", upper_bounds
            # print "lower_bounds:", lower_bounds
            importance = None
            if type(self.replay) == hrl.playback.NearPrioritizedPlayback:
                importance = batch["_weight"]
            info = self.q_function.improve_value_(state=batch["state"], action=batch["action"],
                                                  target_value=np.asarray(target_values),
                                                  td_target_value=np.asarray(targets0),
                                                  importance=importance,
                                                  sess=self.get_session())

            if type(self.replay) == hrl.playback.NearPrioritizedPlayback:
                if "td_losses" in info:
                    self.replay.update_score(index=index, score=info["td_losses"])
                    
            info.update({
                "bounds/lower_bound": np.asarray(lower_bounds),
                "bounds/upper_bound": np.asarray(upper_bounds),
                "bounds/target_value": np.asarray(target_values),
                "bounds/target0": np.asarray(targets0)
            })
        return info

    def get_value(self, state, action=None, **kwargs):
        state = np.asarray(state)
        if list(state.shape) == list(self.state_shape):
            # single sample, make batch
            state = state.reshape([1]+list(self.state_shape))
            if action is not None:
                action = action.reshape([1, self.num_actions])
        kwargs.update({"sess": self.get_session()})
        return self.q_function.get_value(state, action, **kwargs)

        pass

    def new_episode(self, state):
        self.episode_n += 1
        super(OTValueUpdateMixin, self).new_episode(state)


class OTDQN(
    OTValueUpdateMixin,
    hrl.mixin.EpsilonGreedyPolicyMixin,
    hrl.tf_dependent.base.BaseDeepAgent
):
    def __init__(self, state_shape,
                 num_actions,
                 actions,
                 epsilon,
                 batch_size,
                 reward_decay,
                 K,
                 weight_upper_bound,
                 weight_lower_bound,
                 replay_capacity=1000,
                 replay_class=hrl.playback.MapPlayback,
                 priority_bias=0.0,
                 importance_weight=0.0,
                 state_offset_scale=(0, 1),
                 schedule=(1, 10),
                 training_params=(tf.train.AdamOptimizer(0.001), 0.01, 10.0),
                 f_net_dqn=None,
                 **kwargs):
        super(OTDQN, self).__init__(state_shape=state_shape,
                                    num_actions=num_actions,
                                    actions=actions,
                                    epsilon=epsilon,
                                    batch_size=batch_size,
                                    reward_decay=reward_decay,
                                    K=K,
                                    weight_upper_bound=weight_upper_bound,
                                    weight_lower_bound=weight_lower_bound,
                                    replay_capacity=replay_capacity,
                                    replay_class=replay_class,
                                    schedule=schedule,
                                    training_params=training_params,
                                    f_net_dqn=f_net_dqn,
                                    gamma=reward_decay,
                                    state_offset_scale=state_offset_scale,
                                    **kwargs)