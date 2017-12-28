# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl

import hobotrl.network as network
import hobotrl.sampling as sampling
import hobotrl.target_estimate as target_estimate
import hobotrl.tf_dependent.distribution as distribution
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.policy import StochasticPolicy
from value_based import GreedyStateValueFunction
import cv2


class ActorCriticUpdater(network.NetworkUpdater):
    def __init__(self, policy_dist, v_function, target_estimator, entropy=1e-3, actor_weight=1.0):
        """
        Actor Critic methods, for both continuous and discrete action spaces.

        :param policy_dist:
        :type policy_dist: distribution.NNDistribution
        :param v_function: Function calculating state value
        :type v_function: network.NetworkFunction
        :param target_estimator:
        :type target_estimator:
        :param num_actions:
        """
        super(ActorCriticUpdater, self).__init__()
        self._policy_dist, self._v_function = policy_dist, v_function
        self._target_estimator = target_estimator
        self._entropy = entropy
        with tf.name_scope("ActorCriticUpdater"):
            with tf.name_scope("input"):
                self._input_target_v = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_v")
                self._input_action = policy_dist.input_sample()
                self._input_entropy = tf.placeholder(dtype=tf.float32, shape=[], name="input_entropy")
            op_v = v_function.output().op
            with tf.name_scope("value"):
                td = self._input_target_v - op_v
                self._q_loss = tf.reduce_mean(network.Utils.clipped_square(td))
            with tf.name_scope("policy"):
                advantage = self._input_target_v - op_v
                self._advantage = advantage
                _mean, _var = tf.nn.moments(advantage, axes=[0])
                self._std_advantage = advantage / (tf.sqrt(_var) + 1.0)
                # self._std_advantage = self._advantage
                pi_loss = tf.reduce_mean(self._policy_dist.log_prob() * tf.stop_gradient(self._std_advantage))
                entropy_loss = tf.reduce_mean(self._input_entropy * self._policy_dist.entropy())
                self._pi_loss = pi_loss
            self._op_loss = self._q_loss - actor_weight * (self._pi_loss + entropy_loss)
            print "advantage, self._policy_dist.entropy(), self._policy_dist.log_prob()", advantage, self._policy_dist.entropy(), self._policy_dist.log_prob()
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._v_function.variables +
                                                               self._policy_dist._dist_function.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done, next_state1, next_state2, action1, action2, reward1, reward2 = \
            batch["state"][0:-2, :, :, :], \
            batch["action"][0:-2], \
            batch["reward"][0:-2], \
            batch["next_state"][0:-2, :, :, :], \
            batch["episode_done"][0:-2], \
            batch["state"][1:-1, :, :, :], \
            batch["state"][2:, :, :, :], \
            batch["action"][1:-1], \
            batch["action"][2:], \
            batch["reward"][1:-1], \
            batch["reward"][2:]
        target_value = self._target_estimator.estimate(state, action, reward, next_state, episode_done)
        feed_dict = self._v_function.input_dict(state, action, action1, action2)
        feed_dict.update(self._policy_dist.dist_function().input_dict(state, action, action1, action2))
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
        if isinstance(self._policy_dist, hrl.tf_dependent.distribution.NormalDistribution):
            fetch_dict.update({
                "stddev": self._policy_dist.stddev(),
                "mean": self._policy_dist.mean()
            })
        else:
            pass
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)


class PolicyNetUpdater(network.NetworkUpdater):
    def __init__(self, rollout_dist, rollout_action_function, pi_function, entropy=1e-3):
        """
        Policy Net updater
        calculate the loss between action derived from A3C and the policy net

        :param rollout_action:
        :param pi_function:
        :param entropy:
        """
        super(PolicyNetUpdater, self).__init__()
        self._rollout_dist, self._pi_function, self._rollout_action_function = rollout_dist, pi_function, rollout_action_function
        self._entropy = entropy
        with tf.name_scope("ActorCriticUpdater"):
            with tf.name_scope("input"):
                self._input_action = self._rollout_dist.input_sample()

            op_pi = self._pi_function.output().op
            op_mimic_pi = self._rollout_action_function.output().op

            with tf.name_scope("rollout"):
                self._rollout_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=op_pi , logits=op_mimic_pi), name="rollout_loss")
                self._entropy_loss = self._rollout_action_function
            self._op_loss = self._rollout_loss

        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._rollout_action_function.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done, next_state1, next_state2, action1, action2, reward1, reward2 = \
            batch["state"][0:-2, :, :, :], \
            batch["action"][0:-2], \
            batch["reward"][0:-2], \
            batch["next_state"][0:-2, :, :, :], \
            batch["episode_done"][0:-2], \
            batch["state"][1:-1, :, :, :], \
            batch["state"][2:, :, :, :], \
            batch["action"][1:-1], \
            batch["action"][2:], \
            batch["reward"][1:-1], \
            batch["reward"][2:]
        feed_dict = self._rollout_action_function.input_dict(state, action, action1, action2)

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"rollout_loss": self._op_loss})


class EnvModelUpdater(network.NetworkUpdater):
    def __init__(self, next_state_function, reward_function, next_state_function1, reward_function1,
                 next_state_function2, reward_function2, state_shape, entropy=1e-3):
        super(EnvModelUpdater, self).__init__()
        self._next_state_function, self._reward_function = next_state_function, reward_function
        self._next_state_function1, self._reward_function1 = next_state_function1, reward_function1
        self._next_state_function2, self._reward_function2 = next_state_function2, reward_function2
        self._entropy = entropy
        with tf.name_scope("EnvModelUpdater"):
            with tf.name_scope("input"):
                self._input_next_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape),
                                                        name="input_next_state")
                self._input_reward = tf.placeholder(dtype=tf.float32, shape=[None], name="input_reward")

                self._input_next_state1 = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape),
                                                        name="input_next_state1")
                self._input_reward1 = tf.placeholder(dtype=tf.float32, shape=[None], name="input_reward1")

                self._input_next_state2 = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape),
                                                        name="input_next_state2")
                self._input_reward2 = tf.placeholder(dtype=tf.float32, shape=[None], name="input_reward2")

            self.op_next_state = next_state_function.output().op
            self.op_reward = reward_function.output().op

            self.op_next_state1 = next_state_function1.output().op
            self.op_reward1 = reward_function1.output().op

            self.op_next_state2 = next_state_function2.output().op
            self.op_reward2 = reward_function2.output().op

            with tf.name_scope("env_model"):
                self._env_loss = tf.reduce_mean(
                    network.Utils.clipped_square(self.op_next_state - self._input_next_state[:,:,:,9:12]))
                self._reward_loss =tf.reduce_mean(network.Utils.clipped_square(self.op_reward - self._input_reward))

                self._env_loss += tf.reduce_mean(
                    network.Utils.clipped_square(self.op_next_state1 - self._input_next_state1[:, :, :, 9:12]))
                self._reward_loss += tf.reduce_mean(network.Utils.clipped_square(self.op_reward1 - self._input_reward1))

                self._env_loss += tf.reduce_mean(
                    network.Utils.clipped_square(self.op_next_state2 - self._input_next_state2[:, :, :, 9:12]))
                self._reward_loss += tf.reduce_mean(network.Utils.clipped_square(self.op_reward2 - self._input_reward2))

                self._env_loss = self._env_loss / 6.0 * 255.0 * 50.0
                self._reward_loss = self._reward_loss / 6.0

            self._op_loss = self._env_loss + self._reward_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._next_state_function.variables +
                                                               self._reward_function.variables)
        self.imshow_count = 0

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done, next_state1, next_state2, action1, action2, reward1, reward2 = \
            batch["state"][0:-2, :, :, :], \
            batch["action"][0:-2], \
            batch["reward"][0:-2], \
            batch["next_state"][0:-2, :, :, :], \
            batch["episode_done"][0:-2], \
            batch["state"][1:-1, :, :, :], \
            batch["state"][2:, :, :, :], \
            batch["action"][1:-1], \
            batch["action"][2:], \
            batch["reward"][1:-1], \
            batch["reward"][2:]
        feed_dict = self._next_state_function.input_dict(state, action, action1, action2)
        feed_dict.update(self._next_state_function1.input_dict(state, action, action1, action2))
        feed_dict.update(self._next_state_function2.input_dict(state, action, action1, action2))
        feed_more = {
            self._input_next_state: next_state,
            self._input_reward: reward,
            self._input_next_state1: next_state1,
            self._input_reward1: reward1,
            self._input_next_state2: next_state2,
            self._input_reward2: reward2
        }
        feed_dict.update(feed_more)
        self.imshow_count += 1
        print "----------------%s-------------" % self.imshow_count
        if self.imshow_count % 2 == 0:
            width = np.shape(state[0])[1]
            for i in range(len(reward) - 2):
                pred_1 = self._next_state_function(np.reshape(state[i], (1, width, width, 12)), np.reshape(action[i], 1),
                                                   np.reshape(action1[i], 1), np.reshape(action2[i], 1))[0]
                pred_2 = self._next_state_function1(np.reshape(state[i], (1, width, width, 12)), np.reshape(action[i], 1),
                                                   np.reshape(action1[i], 1), np.reshape(action2[i], 1))[0]
                pred_3 = self._next_state_function2(np.reshape(state[i], (1, width, width, 12)), np.reshape(action[i], 1),
                                                   np.reshape(action1[i], 1), np.reshape(action2[i], 1))[0]

                # saving for MsPacman
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_a_raw1.png" % (self.imshow_count,i),
                #             cv2.cvtColor(state[i][:,:,0:3], cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_a_raw2.png" % (self.imshow_count,i),
                #             cv2.cvtColor(state[i][:,:,3:6], cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_a_raw3.png" % (self.imshow_count,i),
                #             cv2.cvtColor(state[i][:,:,6:9], cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_a_raw4.png" % (self.imshow_count,i),
                #             cv2.cvtColor(state[i][:,:,9:12], cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_b_pred_1.png" % (self.imshow_count,i),
                #             cv2.cvtColor(pred_1, cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_d_pred_2.png" % (self.imshow_count,i),
                #             cv2.cvtColor(pred_2, cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_f_pred_3.png" % (self.imshow_count,i),
                #             cv2.cvtColor(pred_3, cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_c_ground_truth_1.png" % (self.imshow_count, i),
                #             cv2.cvtColor(next_state[i][:, :, 9:12], cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_e_ground_truth_2.png" % (self.imshow_count, i),
                #             cv2.cvtColor(next_state[i + 1][:, :, 9:12], cv2.COLOR_RGB2BGR))
                # cv2.imwrite("./log/I2AMsPacman/Img/%s_%s_g_ground_truth_3.png" % (self.imshow_count, i),
                #             cv2.cvtColor(next_state[i + 2][:, :, 9:12], cv2.COLOR_RGB2BGR))

                # saving for CarRacing
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_a_raw1.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * state[i][:, :, 0:3].astype(np.float32), cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_a_raw2.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * state[i][:, :, 3:6].astype(np.float32), cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_a_raw3.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * state[i][:, :, 6:9].astype(np.float32), cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_a_raw4.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * state[i][:, :, 9:12].astype(np.float32), cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_b_pred_1.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * pred_1, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_d_pred_2.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * pred_2, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_f_pred_3.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * pred_3, cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_c_ground_truth_1.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * next_state[i][:, :, 9:12].astype(np.float32), cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_e_ground_truth_2.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * next_state[i + 1][:, :, 9:12].astype(np.float32), cv2.COLOR_RGB2BGR))
                cv2.imwrite("./log/CarRacing/I2ACarRacing_ob_50/Img/%s_%s_g_ground_truth_3.png" % (self.imshow_count, i),
                            cv2.cvtColor(255 * next_state[i + 2][:, :, 9:12].astype(np.float32), cv2.COLOR_RGB2BGR))
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"env_model_loss": self._op_loss,
                                                                  "reward_loss": self._reward_loss,
                                                                  "observation_loss": self._env_loss})


class ActorCriticWithI2A(sampling.TrajectoryBatchUpdate,
          BaseDeepAgent):
    def __init__(self, num_action,
                 f_se, f_ac, f_env, f_rollout, f_encoder, state_shape,
                 # f_se_1, f_se_2, f_se_3, f_se_4, f_ac, f_env, f_rollout, f_encoder, state_shape,
                 # ACUpdate arguments
                 discount_factor, entropy=1e-3, target_estimator=None, max_advantage=10.0,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # sampler arguments
                 sampler=None,
                 batch_size=32,
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
        :param max_advantage: advantage regulation: max advantage value in policy gradient step
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

        def f_iaa(inputs):
            input_observation = inputs[0]
            input_action = inputs[1]
            input_action1 = inputs[2]
            input_action2 = inputs[3]

            se = network.Network([input_observation], f_se, var_scope="se_1")
            se = network.NetworkFunction(se["se"]).output().op

            # se_1 = network.Network([input_observation], f_se_1, var_scope="se_1")
            # se_1 = network.NetworkFunction(se_1["se_1"]).output().op
            #
            # se_2 = network.Network([se_1], f_se_2, var_scope="se_2")
            # se_2 = network.NetworkFunction(se_2["se_2"]).output().op
            #
            # se_3 = network.Network([se_2], f_se_3, var_scope="se_3")
            # se_3 = network.NetworkFunction(se_3["se_3"]).output().op
            #
            # se_4 = network.Network([se_3], f_se_4, var_scope="se_4")
            # se_4 = network.NetworkFunction(se_4["se_4"]).output().op

            input_reward = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="input_reward")
            encode_state = tf.placeholder(dtype=tf.float32, shape=[None, state_shape[0], state_shape[1], 9], name="encode_states")

            rollout = network.Network([input_observation], f_rollout, var_scope="rollout_policy")
            env_model = network.Network([[input_observation], input_action], f_env, var_scope="EnvModel")
            # env_model = network.Network([[se_1], [se_2], [se_3], [se_4], input_action], f_env, var_scope="EnvModel")
            rollout_encoder = network.Network([encode_state, input_reward], f_encoder, var_scope="rollout_encoder")

            current_state = input_observation

            # new added
            # current_rollout = rollout([current_state], name_scope="rollout")
            # rollout_action_function = network.NetworkFunction(current_rollout["rollout_action"])

            # rollout_action_dist = tf.contrib.distributions.Categorical(rollout_action_function.output().op)
            # rollout_action = rollout_action_dist.sample()
            # current_action = tf.one_hot(indices=rollout_action, depth=rollout_action_dist.event_size, on_value=1.0,
            #                             off_value=0.0, axis=-1)

            env_model = env_model([[input_observation], input_action], name_scope="env_model")
            out_next_state = network.NetworkFunction(env_model["next_state"]).output().op
            out_reward = network.NetworkFunction(env_model["reward"]).output().op

            env_model1 = env_model([[
                tf.concat([input_observation[:, :, :, 3:12], out_next_state], axis=3)], input_action1],
                name_scope="env_model1")
            out_next_state1 = network.NetworkFunction(env_model1["next_state"]).output().op
            out_reward1 = network.NetworkFunction(env_model1["reward"]).output().op

            env_model2 = env_model([[
                tf.concat([input_observation[:, :, :, 6:12], out_next_state, out_next_state1], axis=3)], input_action2],
                name_scope="env_model2")
            out_next_state2 = network.NetworkFunction(env_model2["next_state"]).output().op
            out_reward2 = network.NetworkFunction(env_model2["reward"]).output().op

            # out_action = rollout_action_function.output().op
            # out_next_state = next_state
            # out_reward = reward

            for i in range(3):
                for j in range(3):
                    current_rollout = rollout([current_state], name_scope="rollout_%d_%d" %(i,j))
                    rollout_action_function = network.NetworkFunction(current_rollout["rollout_action"])

                    rollout_action_dist = tf.contrib.distributions.Categorical(rollout_action_function.output().op)
                    current_action = rollout_action_dist.sample()
                    # current_action = tf.one_hot(indices=rollout_action, depth=rollout_action_dist.event_size, on_value=1.0, off_value=0.0, axis=-1)

                    env_model = env_model([[current_state], current_action], name_scope="env_model_%d_%d" %(i,j))

                    next_state = network.NetworkFunction(env_model["next_state"]).output().op
                    reward = network.NetworkFunction(env_model["reward"]).output().op

                    if i == 0 and j == 0:
                        out_action = rollout_action_function.output().op
                    if j == 0:
                        # out_next_state = next_state
                        # out_reward = reward
                        encode_states = next_state
                        rollout_reward = reward
                    else:
                        encode_states = tf.concat([next_state, encode_states], axis=3)
                        rollout_reward = tf.concat([rollout_reward, reward], axis=0)

                    current_state = tf.concat([current_state[:,:,:,3:12], next_state], axis=3)

                encode_state = encode_states
                # input_reward = rollout_reward
                input_reward = tf.reshape(rollout_reward, [-1, 3])
                rollout_encoders = rollout_encoder([encode_state, input_reward], name_scope="rollout_encoder_%d" %i)

                re = network.NetworkFunction(rollout_encoders["re"]).output().op
                input_reward = []
                if i == 0:
                    path = re
                else:
                    path = tf.concat([path, re], axis=1)

            feature = tf.concat([path, se], axis=1)

            ac = network.Network([feature], f_ac, var_scope='ac')
            # ac = network.Network([se_4], f_ac, var_scope='ac')
            v = network.NetworkFunction(ac["v"]).output().op
            pi_dist = network.NetworkFunction(ac["pi"]).output().op

            return {"v": v, "pi": pi_dist, "rollout_action": out_action,
                    "next_state": out_next_state, "reward": out_reward,
                    "next_state1": out_next_state1, "reward1": out_reward1,
                    "next_state2": out_next_state2, "reward2": out_reward2}

        kwargs.update({
            "f_iaa": f_iaa,
            "state_shape": state_shape,
            "num_action":num_action,
            "discount_factor": discount_factor,
            "entropy": entropy,
            "target_estimator": target_estimator,
            "max_advantage": max_advantage,
            "max_gradient": max_gradient,
            "batch_size": batch_size,
        })
        print "network_optimizer:", network_optimizer
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        if sampler is None:
            sampler = sampling.TrajectoryOnSampler(interval=batch_size)
            kwargs.update({"sampler": sampler})

        super(ActorCriticWithI2A, self).__init__(*args, **kwargs)
        pi = self.network["pi"]
        if pi is not None:
            # discrete action: pi is categorical probability distribution
            self._pi_function = network.NetworkFunction(self.network["pi"])
            self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")

            self._pi_distribution = distribution.DiscreteDistribution(self._pi_function, self._input_action)
            q = self.network["q"]
            if q is not None:
                # network outputs q
                self._q_function = network.NetworkFunction(q)
                self._v_function = GreedyStateValueFunction(self._q_function)
            else:
                # network output v
                self._v_function = network.NetworkFunction(self.network["v"])
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
            # continuous action: mean / stddev for normal distribution

        self._rollout_action = network.NetworkFunction(self.network["rollout_action"])
        self._rollout_dist = distribution.DiscreteDistribution(self._rollout_action, self._input_action)

        self._next_state_function = network.NetworkFunction(self.network["next_state"])
        self._reward_function = network.NetworkFunction(self.network["reward"])

        self._next_state_function1 = network.NetworkFunction(self.network["next_state1"])
        self._reward_function1 = network.NetworkFunction(self.network["reward1"])

        self._next_state_function2 = network.NetworkFunction(self.network["next_state2"])
        self._reward_function2 = network.NetworkFunction(self.network["reward2"])

        if target_estimator is None:
            target_estimator = target_estimate.NStepTD(self._v_function, discount_factor)
            # target_estimator = target_estimate.GAENStep(self._v_function, discount_factor)
        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(
            ActorCriticUpdater(policy_dist=self._pi_distribution,
                               v_function=self._v_function,
                               target_estimator=target_estimator, entropy=entropy), name="ac")
        network_optimizer.add_updater(network.L2(self.network), name="l2")
        network_optimizer.add_updater(
            PolicyNetUpdater(rollout_dist=self._rollout_dist,
                             rollout_action_function=self._rollout_action,
                             pi_function=self._pi_function),
            name="policy_net"
        )
        network_optimizer.add_updater(
            EnvModelUpdater(next_state_function=self._next_state_function,
                            reward_function=self._reward_function,
                            next_state_function1=self._next_state_function1,
                            reward_function1=self._reward_function1,
                            next_state_function2=self._next_state_function2,
                            reward_function2=self._reward_function2,
                            state_shape=state_shape),
            name="env_model"
        )
        network_optimizer.compile()

        self._policy = StochasticPolicy(self._pi_distribution)

    def init_network(self, f_iaa, state_shape, num_action, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
        input_action1 = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action1")
        input_action2 = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action2")
        return network.Network([input_state, input_action, input_action1, input_action2], f_iaa, var_scope="learn")

    def update_on_trajectory(self, batch):
        if (np.shape(batch["action"])[0] >= 3):
            self.network_optimizer.update("policy_net", self.sess, batch)
            self.network_optimizer.update("env_model", self.sess, batch)
            self.network_optimizer.update("ac", self.sess, batch)
            self.network_optimizer.update("l2", self.sess)
            info = self.network_optimizer.optimize_step(self.sess)
            return info, {}
        else:
            return {}, {}

    def set_session(self, sess):
        super(ActorCriticWithI2A, self).set_session(sess)
        self.network.set_session(sess)
        self._pi_distribution.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)

