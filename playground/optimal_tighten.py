#
# -*- coding: utf-8 -*-

import logging
import sys

import tensorflow as tf
import numpy as np

import hobotrl as hrl
import hobotrl.playback


class OTDQN(
    hrl.mixin.EpsilonGreedyPolicyMixin,
    hrl.tf_dependent.base.BaseDeepAgent):
    def __init__(self,
                 f_net,
                 state_shape,
                 action_n,
                 reward_decay,
                 batch_size,
                 K,
                 bounds_weight,  # weight: λ for upper / lower bounds
                 optimizer,
                 target_sync_interval=100,
                 replay_capacity=10000,
                 replay_class=hrl.playback.MapPlayback,
                 **kwargs):
        super(OTDQN, self).__init__(**kwargs)
        self.state_shape, self.action_n, self.reward_decay, self.batch_size, self.K, \
            self.bounds_weight, self.target_sync_interval = \
            state_shape, action_n, reward_decay, batch_size, K, bounds_weight, target_sync_interval
        self.step_n = 0
        self.replay = replay_class(capacity=replay_capacity, sample_shapes={
            "state": state_shape,
            "action": [],
            "reward": [],
            "next_state": state_shape,
            "episode_done": [],
            "future_reward": [],
            "episode_n": [],
        })
        self.episode_n = 0  # counter for episode
        self.episode_samples = []  # store samples from this episode, for calculating future_reward

        with tf.variable_scope("input"):
            self.input_state = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="input_state")
            self.input_action = tf.placeholder(dtype=tf.int32, shape=[None], name="input_action")
            self.input_next_state = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="input_next_state")
            self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None], name="input_reward")
            self.input_episode_done = tf.placeholder(dtype=tf.float32, shape=[None], name="input_episode_done")
            self.input_target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_v")
            self.input_lower_bound = tf.placeholder(dtype=tf.float32, shape=[None], name="input_lower_bound")
            self.input_upper_bound = tf.placeholder(dtype=tf.float32, shape=[None], name="input_upper_bound")

        with tf.variable_scope("value"):
            with tf.variable_scope("learn"):
                self.q = f_net(self.input_state, action_n)
            with tf.variable_scope("target"):
                self.target_q = f_net(self.input_next_state, action_n)
            with tf.variable_scope("train"):
                target_value = self.input_target_v
                # target_value = tf.reduce_max(self.target_q, axis=1, keep_dims=True)
                current_value = tf.reduce_sum(self.q * tf.one_hot(self.input_action, action_n), axis=1)
                td = self.input_reward + reward_decay * (1.0 - self.input_episode_done) * target_value\
                     - current_value
                td_loss = tf.square(td)
                lower_violated = tf.square(tf.maximum(self.input_lower_bound - current_value, 0))
                upper_violated = tf.square(tf.maximum(current_value - self.input_upper_bound, 0))
                loss = tf.reduce_mean(td_loss) \
                       + bounds_weight * tf.reduce_mean(lower_violated) \
                       + bounds_weight * tf.reduce_mean(upper_violated)

                training_op = optimizer.minimize(loss)
        vars_learn = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="value/learn"
        )
        vars_target = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="value/target"
        )
        print "vars_target, vars_learn:", vars_learn, vars_target
        print "q, target_q, current_value, td, td_loss, loss:", self.q, self.target_q, current_value, td, td_loss, loss
        with tf.name_scope("follow"):
            follows = tf.group(*[tf.assign(target, learn) for target, learn in zip(vars_target, vars_learn)])
        self.td_losses, self.loss, self.training_op, self.follows = td_loss, loss, training_op, follows
        self.lower_losses, self.upper_losses = lower_violated, upper_violated

    def get_value(self, state, **kwargs):
        state = np.asarray(state)
        if list(state.shape) == list(self.state_shape):
            # single sample, make batch
            state = state.reshape([1]+list(self.state_shape))
        return self.sess.run(self.q, feed_dict={self.input_state: state})

    def get_target_value(self, state, **kwargs):
        return self.sess.run(self.target_q, feed_dict={self.input_next_state: np.asarray(state)})

    def update_value(self, state, action, reward, episode_done, target_value, lower_bounds, upper_bounds):
        return self.sess.run([self.training_op, self.td_losses, self.lower_losses, self.upper_losses], feed_dict={
            self.input_state: state,
            self.input_action: action,
            self.input_reward: reward,
            self.input_episode_done: episode_done,
            self.input_target_v: target_value,
            self.input_lower_bound: lower_bounds,
            self.input_upper_bound: upper_bounds
        })

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
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
            # all 'state'
            nearby_states = [[nb_batch['state'] for nb_batch, nb_i in nb_bi] for nb_bi, b_i in nearby_batch_index]
            # last 'next_state'
            nearby_last_next = [[nb_bi[-1][0]['next_state']] for nb_bi, b_i in nearby_batch_index]
            nearby_states = [a + b for a, b in zip(nearby_states, nearby_last_next)]
            state_targets = []
            for n_s in nearby_states:
                t = self.get_target_value(np.asarray(n_s))
                state_targets.append(np.max(t, axis=1))
            # print "nearby_states:", nearby_states
            # print "state_targets:", state_targets
            # print "nearby_batch_index", nearby_batch_index
            lower_bounds, upper_bounds, target_values = [], [], []
            for _i in range(len(nearby_batch_index)):
                nb_bi, sample_i = nearby_batch_index[_i]
                targets = state_targets[_i]
                low, high = batch["future_reward"][_i], sys.maxint
                nb_indices = [x[1] for x in nb_bi]
                sample_index = nb_indices.index(sample_i)
                # print "sample_i, nb_indices, sample_index:", sample_i, nb_indices, sample_index
                r = 0.0
                for k in range(1, self.K+1):  # upper bound: k -> [1, K]
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
                for k in range(1, self.K+1):
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
                low, high = batch["future_reward"][_i], sys.maxint
                lower_bounds.append(low)
                upper_bounds.append(high)
                target_values.append(targets[sample_index+1])
            # print "target_values:", target_values
            # print "upper_bounds:", upper_bounds
            # print "lower_bounds:", lower_bounds

            _, td_losses, lower_loss, upper_loss = self.update_value(state=batch["state"],
                                                                     action=batch["action"],
                                                                     reward=batch["reward"],
                                                                     episode_done=np.asarray(batch["episode_done"]),
                                                                     target_value=np.asarray(target_values),
                                                                     upper_bounds=np.asarray(upper_bounds),
                                                                     lower_bounds=np.asarray(lower_bounds))
            info["td_losses"] = td_losses
            info.update({
                "td_losses": td_losses,
                "L_losses": lower_loss,
                "U_losses": upper_loss
            })
        if self.step_n % self.target_sync_interval == 0:
            self.get_session().run(self.follows)
        super_info = super(OTDQN, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        super_info.update(info)
        return super_info

    def new_episode(self, state):
        self.episode_n += 1
        super(OTDQN, self).new_episode(state)

