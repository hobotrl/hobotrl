# -*- coding: utf-8 -*-

import os
import operator
import logging
import tensorflow as tf
import numpy as np
import copy
import hobotrl as hrl

import hobotrl.network as network
import hobotrl.sampling as sampling
import hobotrl.target_estimate as target_estimate
import hobotrl.tf_dependent.distribution as distribution
import hobotrl.utils as utils
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.policy import StochasticPolicy
from value_based import GreedyStateValueFunction
import cv2
from hobotrl.cvutils import flow_to_color


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
        state, action, reward, next_state, episode_done = \
            batch["state"], \
            batch["action"], \
            batch["reward"], \
            batch["next_state"], \
            batch["episode_done"]
        target_value = self._target_estimator.estimate(state, action, reward, next_state, episode_done)
        feed_dict = self._v_function.input_dict(state, action)
        feed_dict.update(self._policy_dist.dist_function().input_dict(state, action))
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
        with tf.name_scope("PolicyNetUpdater"):
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
        state, action, reward, next_state, episode_done = \
            batch["state"], \
            batch["action"], \
            batch["reward"], \
            batch["next_state"], \
            batch["episode_done"]
        feed_dict = self._rollout_action_function.input_dict(state, action)

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"rollout_loss": self._op_loss})


class EnvModelUpdater(network.NetworkUpdater):
    def __init__(self, net_se, net_transition, net_decoder, state_shape, dim_action,
                 curriculum=None, skip_step=None, transition_weight=0.0, with_momentum=True, compute_with_diff=False,
                 save_image_interval=1000, detailed_decoder=False):
        super(EnvModelUpdater, self).__init__()
        if curriculum is None:
            self._curriculum = [1, 3, 5]
            self._skip_step = [5000, 15000]
        else:
            self._curriculum = curriculum
            self._skip_step = skip_step

        self._depth = self._curriculum[-1]
        self.save_image_interval = save_image_interval
        self._detailed_decoder = detailed_decoder
        with tf.name_scope("EnvModelUpdater"):
            with tf.name_scope("input"):
                self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None],
                                                    name="input_action")
                self._input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape),
                                                   name="input_state")
                self._input_reward = tf.placeholder(dtype=tf.float32, shape=[None], name="input_reward")
                self._count = tf.placeholder(dtype=tf.int32, name="count")

            with tf.name_scope("inputs"):
                s0 = self._input_state[:-1]
                state_shape = tf.shape(self._input_state)[1:]

                f0 = s0[:, :, :, -3:]
                logging.warning("s0:%s, f0:%s", s0.shape, f0.shape)
                sn, an, rn, fn =[], [], [], []
                for i in range(self._depth):
                    sn.append(self._input_state[i+1:])
                    an.append(self._input_action[i:])
                    rn.append(self._input_reward[i:])
                    fn.append(sn[-1][:, :, :, -3:])

            with tf.name_scope("rollout"):
                ses_predict = []
                goalfrom0_predict = []
                momfrom0_predict = []
                action_relatedfrom0_predict = []
                r_predict = []
                r_predict_loss = []
                f_predict = []
                image_channel = None
                f_predict_loss = []
                transition_loss = []
                momentum_loss = []
                mom_decoder_predict = []
                action_related_decoder_predict = []
                if compute_with_diff:
                    diff_ob = []
                    for i in range(self._input_state.shape[-1] / 3 - 1):
                        diff_ob.append(
                            self._input_state[:, :, :, (i + 1) * 3:(i + 1) * 3 + 3] - self._input_state[:, :, :,
                                                                                      i * 3:i * 3 + 3])
                    ses = net_se([tf.concat(diff_ob[:], axis=3)])["se"].op
                else:
                    ses = net_se([self._input_state])["se"].op
                se0 = ses[:-1]
                sen = []
                for i in range(self._depth):
                    sen.append(ses[i+1:])
                cur_se = se0
                cur_goal = None
                cur_mom = None
                cur_action_related = None
                se0_truncate, f0_truncate = se0, f0
                flows = []
                flow_regulations = []
                for i in range(self._depth):
                    logging.warning("[%s]: state:%s, action:%s", i, cur_se.shape, an[i].shape)
                    input_action = tf.one_hot(indices=an[i], depth=dim_action, on_value=1.0, off_value=0.0,
                                              axis=-1)
                    net_trans = net_transition([cur_se, input_action], name_scope="transition_%d" % i)

                    if with_momentum:
                        TM_goal = net_trans["momentum"].op
                        action_related = net_trans["action_related"].op
                        cur_mom = TM_goal if cur_goal is None else cur_goal + TM_goal
                        momfrom0_predict.append(cur_mom)
                        cur_action_related = action_related if cur_goal is None else cur_goal + action_related
                        action_relatedfrom0_predict.append(cur_action_related)
                        cur_se_mom = se0_truncate + cur_mom
                        cur_se_action_related = se0_truncate + cur_action_related
                        momentum_loss.append(tf.reduce_mean(network.Utils.clipped_square(cur_se_mom - sen[i])))

                    goal = net_trans["next_state"].op
                    # socalled_state = net_trans["action_related"].op
                    cur_goal = goal if cur_goal is None else tf.stop_gradient(cur_goal) + goal
                    goalfrom0_predict.append(cur_goal)
                    cur_se = se0_truncate + cur_goal
                    # cur_se = socalled_state

                    ses_predict.append(cur_se)
                    r_predict.append(net_trans["reward"].op)
                    r_predict_loss.append(tf.reduce_mean(network.Utils.clipped_square(r_predict[-1] - rn[i])))
                    # f_predict.append(net_decoder([tf.concat([se0, cur_goal], axis=1), f0],
                    #                              name_scope="frame_decoder%d" % i)["next_frame"].op)
                    if detailed_decoder:
                        mom_decoder_predict.append(net_decoder([tf.concat([se0_truncate, cur_se_mom], axis=1), f0_truncate],
                                                               name_scope="mom_decoder%d" % i)["next_frame"].op)
                        action_related_decoder_predict.append(net_decoder([tf.concat([se0_truncate, cur_se_action_related], axis=1), f0_truncate],
                                                              name_scope="action_related_decoder%d" % i)["next_frame"].op)

                    net_decoded = net_decoder([tf.concat([se0_truncate, cur_goal], axis=1), f0_truncate],
                                              name_scope="frame_decoder%d" % i)
                    f_predict.append(net_decoded["next_frame"].op)
                    predicted_channel = net_decoded["image_channel"]
                    if predicted_channel is not None and image_channel is None:
                        image_channel = predicted_channel.op
                    frame_2 = net_decoded["frame_2"]
                    frame_losses = []
                    if frame_2 is not None:
                        sub_i = 1
                        while True:
                            sub = "frame_%d" % (2**sub_i)
                            sub_frame = net_decoded[sub]
                            if sub_frame is None:
                                break
                            sub_frame = sub_frame.op
                            frame_losses.append(tf.reduce_mean(network.Utils.clipped_square(
                                sub_frame - tf.image.resize_images(fn[i], sub_frame.shape.as_list()[1:3]))
                            ))
                            sub_i = sub_i + 1
                    flow = net_decoded["flow"]
                    if flow is not None:
                        flow = flow.op
                        flows.append(flow)
                        o1_y = flow[:, :-1, :, :] - flow[:, 1:, :, :]
                        o2_y = o1_y[:, :-1, :, :] - o1_y[:, 1:, :, :]
                        o1_x = flow[:, :, :-1, :] - flow[:, :, 1:, :]
                        o2_x = o1_x[:, :, :-1, :] - o1_x[:, :, 1:, :]
                        l1_y = tf.reduce_mean(tf.abs(o2_y))
                        l1_x = tf.reduce_mean(tf.abs(o2_x))
                        flow_regulations.append(l1_x + l1_y)

                    frame_losses.append(tf.reduce_mean(network.Utils.clipped_square(f_predict[-1] - fn[i])))
                    f_predict_loss.append(frame_losses)
                    transition_loss.append(tf.reduce_mean(network.Utils.clipped_square(ses_predict[-1] - sen[i])))
                    cur_goal = cur_goal[:-1]
                    cur_se = cur_se[:-1]
                    f0_truncate = f0_truncate[:-1]
                    se0_truncate = se0_truncate[:-1]

                self._reward_loss = []
                self._env_loss = []
                self._transition_loss = []
                self._momentum_loss = []
                self._flow_regulation_loss = []
                for i in range(len(curriculum)):
                    self._reward_loss.append(tf.reduce_mean(tf.add_n(
                        r_predict_loss[0:curriculum[i]]) / float(curriculum[i]),
                                                            name="reward_loss%d" % curriculum[i]) / 2.0)

                    self._env_loss.append(tf.reduce_mean(tf.add_n(
                        reduce(operator.add, f_predict_loss[0:curriculum[i]], [])) / float(curriculum[i]),
                                                         name="env_loss%d" % curriculum[i]) / 2.0 * 255.0)

                    self._transition_loss.append(tf.reduce_mean(tf.add_n(
                        transition_loss[0:curriculum[i]]) / float(curriculum[i]),
                                                                name="transition_loss%d" % curriculum[i]))

                    if with_momentum:
                        self._momentum_loss.append(tf.reduce_mean(tf.add_n(
                            momentum_loss[0:curriculum[i]]) / float(curriculum[i]),
                                                                  name="momentum_loss%d" % curriculum[i]))
                    else:
                        self._momentum_loss.append(0)

                    if len(flow_regulations) > 0:
                        self._flow_regulation_loss.append(tf.reduce_mean(tf.add_n(
                            flow_regulations[0:curriculum[i]]) / float(curriculum[i]),
                                                                         name="flow_loss%d" % curriculum[i]) * 1e-1)
                    else:
                        self._flow_regulation_loss.append(0.0)

                def loss_assign(index):
                    return tf.gather(self._env_loss, index), \
                           tf.gather(self._reward_loss, index), \
                           tf.gather(self._transition_loss, index), \
                           tf.gather(self._momentum_loss, index), \
                           tf.gather(self._flow_regulation_loss, index), \
                           self._count

                self._env_loss, self._reward_loss, self._transition_loss, self._momentum_loss, \
                self._flow_regulation_loss, self._num = loss_assign(tf.where(tf.equal(self._curriculum, self._count)))

                self._op_loss = self._env_loss \
                                + self._reward_loss \
                                + self._transition_loss \
                                + self._momentum_loss \
                                + self._flow_regulation_loss

            self._s0, self._f0, self._fn, self._f_predict = s0, f0, fn, f_predict
            self._mom_decoder_predict, self._action_related_decoder_predict = \
                mom_decoder_predict, action_related_decoder_predict

            self._flows = flows
            self._image_channel = image_channel
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=net_transition.variables + net_se.variables +
                                                      net_decoder.variables)
        self.imshow_count = 693687
        self.num = 1

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state = batch["state"], batch["action"], batch["reward"], batch["next_state"]
        state = np.concatenate((state, next_state[-1:]), axis=0)

        if len(self._skip_step) == 0:
            self.num = self._curriculum[0]
        else:
            for j in range(len(self._skip_step)):
                if self.imshow_count < self._skip_step[j]:
                    self.num = self._curriculum[j]
                    break
            if self.imshow_count >= self._skip_step[-1]:
                self.num = self._curriculum[-1]

        feed_dict = {
            self._input_state: state,
            self._input_action: action,
            self._input_reward: reward,
            self._count: self.num
        }
        self.imshow_count += 1
        logging.warning("----------------%s minibatches-------------" % self.imshow_count)
        fetch_dict = {"env_model_loss": self._op_loss,
                      "reward_loss": self._reward_loss,
                      "observation_loss": self._env_loss,
                      "transition_loss": self._transition_loss,
                      "momentum_loss": self._momentum_loss,
                      "num": self._num,
                      "flow_regulation_loss": self._flow_regulation_loss
                      }
                      #,
                      # "goal_reg_loss": self._goal_reg_loss}
        if self.imshow_count % self.save_image_interval == 0:
            fetch_dict["s0"] = self._s0
            fetch_dict["update_step"] = self.imshow_count
            for i in range(self.num):
                fetch_dict.update({
                    "f%d" % i: self._fn[i],
                    "f%d_predict" % i: self._f_predict[i],
                })
                if self._detailed_decoder:
                    fetch_dict.update({
                        "a%d_predict" % i: self._action_related_decoder_predict[i],
                        "m%d_predict" % i: self._mom_decoder_predict[i]
                    })
                if len(self._flows) > 0:
                    fetch_dict.update({
                        "flow%d" % i: self._flows[i]
                    })
                if self._image_channel is not None:
                    fetch_dict.update({
                        "image_channel": self._image_channel
                    })
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)

    @staticmethod
    def check_save_image(updater_name, info, log_dir):
        prefix = "EnvModelUpdater/%s/" % updater_name

        num = info[prefix + "num"]
        logging.warning("-----------%s steps for loss------------", num)
        if prefix + "s0" in info:
            s0 = info[prefix + "s0"]
            update_step = info[prefix + "update_step"]
            path_prefix = os.sep.join([log_dir, "Img", ""])
            if not os.path.isdir(path_prefix):
                os.makedirs(path_prefix)
            logging.warning("writing images to %s", path_prefix)
            for i in range(len(s0)):
                s = s0[i]
                frame_n = s.shape[-1] / 3
                for j in range(frame_n):
                    f = s[:, :, 3 * j: 3 * j + 3]
                    cv2.imwrite(path_prefix + "%d_%03d_f0_%d.png" % (update_step, i, j),
                                cv2.cvtColor(255 * f.astype(np.float32), cv2.COLOR_RGB2BGR))
            for d in range(num):
                for i in range(len(s0) - d):
                    fn = info[prefix + "f%d" % d][i]
                    fn_predict = info[prefix + "f%d_predict" % d][i]
                    an_predict, mn_predict = None, None
                    if prefix + "a%d_predict" % d in info:
                        an_predict = info[prefix + "a%d_predict" % d][i]
                    if prefix + "m%d_predict" % d in info:
                        mn_predict = info[prefix + "m%d_predict" % d][i]
                    flow = None
                    if prefix + "flow0" in info:
                        flow = info[prefix + "flow%d" % d][i]
                    # logging.warning("---------------------------------")
                    # logging.warning(np.mean(fn_predict))
                    # logging.warning(np.mean(an_predict))
                    # logging.warning(np.mean(mn_predict))

                    cv2.imwrite(path_prefix + "%d_%03d_f%d_raw.png" % (update_step, i, d + 1),
                                cv2.cvtColor(255 * fn.astype(np.float32), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(path_prefix + "%d_%03d_f%d_predict.png" % (update_step, i, d + 1),
                                cv2.cvtColor(255 * fn_predict.astype(np.float32), cv2.COLOR_RGB2BGR))
                    if an_predict is not None:
                        cv2.imwrite(path_prefix + "%d_%03d_y_actionf%d_predict.png" % (update_step, i, d + 1),
                                    cv2.cvtColor(255 * an_predict.astype(np.float32), cv2.COLOR_RGB2BGR))
                    if mn_predict is not None:
                        cv2.imwrite(path_prefix + "%d_%03d_z_momf%d_predict.png" % (update_step, i, d + 1),
                                    cv2.cvtColor(255 * mn_predict.astype(np.float32), cv2.COLOR_RGB2BGR))
                    if flow is not None:
                        for channel in range(flow.shape[-1] / 2):
                            f = flow[:, :, 2*channel:2*channel+2]
                            cv2.imwrite(path_prefix + "%d_%03d_g_flow%d_%d.png" % (update_step, i, d + 1, channel),
                                        flow_to_color(f.astype(np.float32), max_len=16.0) * 255)
                    if d == 0 and prefix + "image_channel" in info:
                        image_channel = info[prefix + "image_channel"][i]
                        logging.warning("image_channel: %s", image_channel.shape)
                        for channel in range(image_channel.shape[-1] / 3):
                            image = image_channel[:, :, 3*channel:3*channel+3]
                            cv2.imwrite(path_prefix + "%d_%03d_g_channels_%d.png" % (update_step, i, channel),
                                        cv2.cvtColor(255 * image.astype(np.float32), cv2.COLOR_RGB2BGR))

            del info[prefix + "s0"]
            del info[prefix + "update_step"]
            for d in range(num):
                del info[prefix + "f%d" % d], info[prefix + "f%d_predict" % d]
                if prefix + "a%d_predict" % d in info:
                    del info[prefix + "a%d_predict" % d]
                if prefix + "m%d_predict" % d in info:
                    del info[prefix + "m%d_predict" % d]
                if prefix + "flow0" in info:
                    del info[prefix + "flow%d" % d]


class ActorCriticWithI2A(sampling.TrajectoryBatchUpdate,
          BaseDeepAgent):
    def __init__(self, num_action,
                 f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, state_shape,
                 # ACUpdate arguments
                 discount_factor, entropy=1e-3, target_estimator=None, max_advantage=10.0,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # sampler arguments
                 sampler=None,
                 policy_with_iaa=False,
                 compute_with_diff=False,
                 with_momentum=True,
                 rollout_depth=3,
                 rollout_lane=3,
                 dynamic_rollout=None,
                 dynamic_skip_step=None,
                 model_train_depth=3,
                 batch_size=32,
                 save_image_interval=1000,
                 log_dir="./log/img",
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
        :param max_rollout: optional, should be an odd number
        :param args:
        :param kwargs:
        """

        self.processed_state_shape = []
        def f_iaa(inputs):
            input_observation = inputs[0]
            if compute_with_diff:
                logging.warning("use diff 2333")
                diff_ob = []
                for i in range(input_observation.shape[-1] / 3 - 1):
                    diff_ob.append(input_observation[:, :, :, (i+1)*3:(i+1)*3+3] - input_observation[:, :, :, i*3:i*3+3])
                net_se = network.Network([tf.concat(diff_ob[:], axis=3)], f_se, var_scope="se_1")
                self.processed_state_shape = copy.copy(state_shape)
                self.processed_state_shape[-1] = state_shape[-1] - 3
            else:
                net_se = network.Network([input_observation], f_se, var_scope="se_1")
                self.processed_state_shape = state_shape
            input_action = inputs[1]
            action_dim = inputs[2]
            input_action = tf.one_hot(indices=input_action, depth=action_dim, on_value=1.0, off_value=0.0, axis=-1)

            se = net_se["se"].op

            input_reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_reward")
            encode_state = tf.placeholder(dtype=tf.float32, shape=[None, se.shape.as_list()[-1]],
                                          name="encode_states")
            input_frame = tf.placeholder(dtype=tf.float32, shape=[None, state_shape[0], state_shape[1], 3],
                                         name="input_frame")
            rollout = network.Network([se], f_rollout, var_scope="rollout_policy")
            net_model = network.Network([se, input_action], f_tran, var_scope="TranModel")

            net_decoder = network.Network([tf.concat((se, se), axis=-1), input_frame], f_decoder, var_scope="Decoder")

            rollout_encoder = network.Network([tf.concat((encode_state, encode_state, input_reward), axis=-1)],
                                              f_encoder, var_scope="rollout_encoder")

            current_state = se

            for i in range(rollout_lane):
                for j in range(rollout_depth):
                    current_rollout = rollout([current_state], name_scope="rollout_%d_%d" %(i,j))

                    # rollout_action_dist = tf.contrib.distributions.Categorical(rollout_action_function.output().op)
                    # current_action = rollout_action_dist.sample()

                    tran_model = net_model([current_state, current_rollout["rollout_action"].op],
                                           name_scope="env_model_%d_%d" %(i,j))

                    next_goal = tran_model["next_state"].op
                    reward = tran_model["reward"].op

                    if j == 0:
                        encode_states = next_goal
                        rollout_reward = reward
                    else:
                        encode_states = tf.concat([next_goal, encode_states], axis=1)
                        rollout_reward = tf.concat([rollout_reward, reward], axis=0)

                    current_state += next_goal

                current_state = se

                encode_state = tf.split(encode_states, rollout_depth, axis=1)
                input_reward = tf.reshape(rollout_reward, [-1, rollout_depth])
                input_reward = tf.split(input_reward, rollout_depth, axis=1)

                for m in range(rollout_depth):
                    if m == 0:
                        rollout_encoder = rollout_encoder([
                            tf.concat([encode_state[-(m + 1)], encode_state[-(m + 1)], input_reward[-(m + 1)]],
                                      axis=-1)], name_scope="rollout_encoder_%d_%d" %(i, m))
                        re = rollout_encoder["re"].op

                    else:
                        rollout_encoder = rollout_encoder([
                            tf.concat([re, encode_state[-(m + 1)], input_reward[-(m + 1)]],
                                      axis=-1)], name_scope="rollout_encoder_%d_%d" % (i, m))
                        re = rollout_encoder["re"].op

                if i == 0:
                    path = re
                else:
                    path = tf.concat([path, re], axis=1)
            if policy_with_iaa:
                feature = tf.concat([path, se], axis=1)
            else:
                feature = se
            ac = network.Network([feature], f_ac, var_scope='ac')
            v = ac["v"].op
            pi_dist = ac["pi"].op

            return {"v": v, "pi": pi_dist, "rollout_action": None}, \
                    {
                        "se": net_se, "transition": net_model,
                        "state_decoder": net_decoder
                    }
        self._log_dir = log_dir
        self._rollout_depth = rollout_depth
        if dynamic_rollout is None:
            self._dynamic_rollout = [1, 3, 5]
            self._dynamic_skip_step = [5000, 15000]
        else:
            self._dynamic_rollout = dynamic_rollout
            self._dynamic_skip_step = dynamic_skip_step
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
        logging.warning(network_optimizer)
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

        # self._rollout_action = network.NetworkFunction(self.network["rollout_action"])
        # self._rollout_dist = distribution.DiscreteDistribution(self._rollout_action, self._input_action)

        if target_estimator is None:
            target_estimator = target_estimate.NStepTD(self._v_function, discount_factor)
            # target_estimator = target_estimate.GAENStep(self._v_function, discount_factor)
        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(
            ActorCriticUpdater(policy_dist=self._pi_distribution,
                               v_function=self._v_function,
                               target_estimator=target_estimator, entropy=entropy), name="ac")
        network_optimizer.add_updater(network.L2(self.network), name="l2")
        # network_optimizer.add_updater(
        #     PolicyNetUpdater(rollout_dist=self._rollout_dist,
        #                      rollout_action_function=self._rollout_action,
        #                      pi_function=self._pi_function),
        #     name="policy_net"
        # )
        network_optimizer.add_updater(
            EnvModelUpdater(
                net_se=self.network.sub_net("se"),
                net_transition=self.network.sub_net("transition"),
                net_decoder=self.network.sub_net("state_decoder"),
                curriculum=self._dynamic_rollout,
                skip_step=self._dynamic_skip_step,
                state_shape=state_shape,
                dim_action=num_action,
                transition_weight=1.0,
                with_momentum=with_momentum,
                compute_with_diff=compute_with_diff,
                save_image_interval=save_image_interval
            ),
            name="env_model")
        # network_optimizer.freeze(self.network.sub_net("transition").variables)
        network_optimizer.compile()

        self._policy = StochasticPolicy(self._pi_distribution)

    def init_network(self, f_iaa, state_shape, num_action, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
        return network.Network([input_state, input_action, num_action], f_iaa, var_scope="learn")

    def update_on_trajectory(self, batch):
        if (np.shape(batch["action"])[0] >= self._dynamic_rollout[-1]):
            # self.network_optimizer.update("policy_net", self.sess, batch)
            self.network_optimizer.update("env_model", self.sess, batch)
            self.network_optimizer.update("ac", self.sess, batch)
            self.network_optimizer.update("l2", self.sess)
            info = self.network_optimizer.optimize_step(self.sess)
            EnvModelUpdater.check_save_image("env_model", info, self._log_dir)
            return info, {}
        else:
            return {}, {}

    def set_session(self, sess):
        super(ActorCriticWithI2A, self).set_session(sess)
        self.network.set_session(sess)
        self._pi_distribution.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)

