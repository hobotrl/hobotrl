# -*- coding: utf-8 -*-

import os
import logging
import tensorflow as tf
import numpy as np
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
                 depth=5, transition_weight=0.0, with_momentum=True):
        super(EnvModelUpdater, self).__init__()
        self._depth = depth
        with tf.name_scope("EnvModelUpdater"):
            with tf.name_scope("input"):
                self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None],
                                                    name="input_action")
                self._input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape),
                                                   name="input_state")
                self._input_reward = tf.placeholder(dtype=tf.float32, shape=[None], name="input_reward")
                self._count = tf.placeholder(dtype=tf.int32, name="count")

            with tf.name_scope("inputs"):
                s0 = self._input_state[:-depth]
                state_shape = tf.shape(self._input_state)[1:]

                f0 = s0[:, :, :, -3:]
                logging.warning("s0:%s, f0:%s", s0.shape, f0.shape)
                sn, an, rn, fn =[], [], [], []
                for i in range(depth):
                    if i < depth - 1:
                        sn.append(self._input_state[i+1:i-depth+1])
                        an.append(self._input_action[i:i-depth+1])
                        rn.append(self._input_reward[i:i-depth+1])
                    else:
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
                f_predict_loss = []
                transition_loss = []
                momentum_loss = []
                mom_decoder_predict = []
                action_related_decoder_predict = []
                ses = net_se([self._input_state])["se"].op
                se0 = ses[:-depth]
                sen = []
                for i in range(depth):
                    if i < depth - 1:
                        sen.append(ses[i+1:i-depth+1])
                    else:
                        sen.append(ses[i+1:])
                cur_se = se0
                cur_goal = None
                cur_mom = None
                cur_action_related = None

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
                        cur_se_mom = se0 + cur_mom
                        cur_se_action_related = se0 + cur_action_related
                        momentum_loss.append(network.Utils.clipped_square(cur_se_mom - sen[i]))

                    goal = net_trans["next_state"].op
                    # socalled_state = net_trans["action_related"].op
                    cur_goal = goal if cur_goal is None else tf.stop_gradient(cur_goal) + goal
                    goalfrom0_predict.append(cur_goal)
                    cur_se = se0 + cur_goal
                    # cur_se = socalled_state

                    ses_predict.append(cur_se)
                    r_predict.append(net_trans["reward"].op)
                    r_predict_loss.append(network.Utils.clipped_square(r_predict[-1] - rn[i]))
                    # f_predict.append(net_decoder([tf.concat([se0, cur_goal], axis=1), f0],
                    #                              name_scope="frame_decoder%d" % i)["next_frame"].op)
                    mom_decoder_predict.append(net_decoder([tf.concat([se0, cur_se_mom], axis=1), f0],
                                                           name_scope="mom_decoder%d" % i)["next_frame"].op)
                    action_related_decoder_predict.append(net_decoder([tf.concat([se0, cur_se_action_related], axis=1), f0],
                                                          name_scope="action_related_decoder%d" % i)["next_frame"].op)

                    net_decoded = net_decoder([tf.concat([se0, cur_se], axis=1), f0],
                                              name_scope="frame_decoder%d" % i)
                    f_predict.append(net_decoded["next_frame"].op)
                    frame_2 = net_decoded["frame_2"]
                    if frame_2 is not None:
                        frame_2 = net_decoded["frame_2"].op
                        f_predict_loss.append(tf.reduce_mean(network.Utils.clipped_square(
                            frame_2 - tf.image.resize_images(fn[i], frame_2.shape.as_list()[1:3]))
                        ))
                        frame_4 = net_decoded["frame_4"].op
                        f_predict_loss.append(tf.reduce_mean(network.Utils.clipped_square(
                            frame_4 - tf.image.resize_images(fn[i], frame_4.shape.as_list()[1:3]))
                        ))
                        frame_8 = net_decoded["frame_8"].op
                        f_predict_loss.append(tf.reduce_mean(network.Utils.clipped_square(
                            frame_8 - tf.image.resize_images(fn[i], frame_8.shape.as_list()[1:3]))
                        ))
                        flow = net_decoded["flow"].op
                        flows.append(flow)

                        o1_y = flow[:, :-1, :, :] - flow[:, 1:, :, :]
                        o2_y = o1_y[:, :-1, :, :] - o1_y[:, 1:, :, :]
                        o1_x = flow[:, :, :-1, :] - flow[:, :, 1:, :]
                        o2_x = o1_x[:, :, :-1, :] - o1_x[:, :, 1:, :]
                        l1_y = tf.reduce_mean(tf.abs(o2_y))
                        l1_x = tf.reduce_mean(tf.abs(o2_x))
                        flow_regulations.append(l1_x + l1_y)
                    f_predict_loss.append(tf.reduce_mean(network.Utils.clipped_square(f_predict[-1] - fn[i])))
                    transition_loss.append(network.Utils.clipped_square(ses_predict[-1] - sen[i]))

                if len(flow_regulations) > 0:
                    self._flow_regulation_loss = tf.reduce_mean(tf.add_n(flow_regulations) / depth, name="flow_loss") * 1e-1
                else:
                    self._flow_regulation_loss = 0

                self._reward_loss5 = tf.reduce_mean(tf.add_n(r_predict_loss) / depth, name="reward_loss5")
                self._reward_loss3 = tf.reduce_mean(tf.add_n(r_predict_loss[0:3]) / 3.0, name="reward_loss3")
                self._reward_loss1 = tf.reduce_mean(r_predict_loss[0], name="reward_loss1")

                self._env_loss5 = tf.reduce_mean(tf.add_n(f_predict_loss) / depth, name="env_loss5")
                self._env_loss3 = tf.reduce_mean(tf.add_n(f_predict_loss[0:3]) / 3.0, name="env_loss3")
                self._env_loss1 = tf.reduce_mean(f_predict_loss[0], name="env_loss1")

                self._transition_loss5 = tf.reduce_mean(tf.add_n(transition_loss) / depth, name="transition_loss5") \
                                         * transition_weight
                self._transition_loss3 = tf.reduce_mean(tf.add_n(transition_loss[0:3]) / 3.0, name="transition_loss3") \
                                         * transition_weight
                self._transition_loss1 = tf.reduce_mean(transition_loss[0], name="transition_loss1") \
                                         * transition_weight

                if with_momentum:
                    self._momentum_loss5 = tf.reduce_mean(tf.add_n(momentum_loss) / depth, name="momentum_loss5")
                    self._momentum_loss3 = tf.reduce_mean(tf.add_n(momentum_loss[0:3]) / 3.0, name="momentum_loss3")
                    self._momentum_loss1 = tf.reduce_mean(momentum_loss[0], name="momentum_loss1")
                else:
                    self._momentum_loss5 = 0
                    self._momentum_loss3 = 0
                    self._momentum_loss1 = 0
                self._env_loss5 = self._env_loss5 / 2.0 * 255
                self._reward_loss5 = self._reward_loss5 / 2.0
                self._env_loss3 = self._env_loss3 / 2.0 * 255
                self._reward_loss3 = self._reward_loss3 / 2.0
                self._env_loss1 = self._env_loss1 / 2.0 * 255
                self._reward_loss1 = self._reward_loss1 / 2.0

                def f1():
                    return self._env_loss1, self._reward_loss1, self._transition_loss1, self._momentum_loss1, 1

                def f3():
                    return self._env_loss3, self._reward_loss3, self._transition_loss3, self._momentum_loss3, 3

                def f5():
                    return self._env_loss5, self._reward_loss5, self._transition_loss5, self._momentum_loss5, 5

                self._env_loss, self._reward_loss, self._transition_loss, self._momentum_loss, self._num = tf.case({
                    tf.greater(self._count, tf.constant(25000)): f5, tf.less(self._count, tf.constant(10000)): f1},
                    default=f3, exclusive=True)

                self._op_loss = self._env_loss \
                                + self._reward_loss \
                                + self._transition_loss \
                                + self._momentum_loss \
                                + self._flow_regulation_loss

            self._s0, self._f0, self._fn, self._f_predict = s0, f0, fn, f_predict
            self._mom_decoder_predict, self._action_related_decoder_predict = \
                mom_decoder_predict, action_related_decoder_predict

            self._flows = flows
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=net_transition.variables + net_se.variables +
                                                      net_decoder.variables)
        self.imshow_count = 0

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state = batch["state"], batch["action"], batch["reward"], batch["next_state"]
        state = np.concatenate((state, next_state[-1:]), axis=0)
        feed_dict = {
            self._input_state: state,
            self._input_action: action,
            self._input_reward: reward,
            self._count: self.imshow_count
        }
        self.imshow_count += 1
        logging.warning("----------------%s episodes-------------" % self.imshow_count)
        fetch_dict = {"env_model_loss": self._op_loss,
                      "reward_loss": self._reward_loss,
                      "observation_loss": self._env_loss,
                      "transition_loss": self._transition_loss,
                      "momentum_loss": self._momentum_loss,
                      "num": self._num,
                      "flow_regulation_loss": self._flow_regulation_loss
                      }#,
                      # "goal_reg_loss": self._goal_reg_loss}
        if self.imshow_count % 2 == 0:
            fetch_dict["s0"] = self._s0
            fetch_dict["update_step"] = self.imshow_count
            for i in range(self._depth):
                fetch_dict.update({
                    "f%d" % i: self._fn[i],
                    "f%d_predict" % i: self._f_predict[i],
                    "a%d_predict" % i: self._action_related_decoder_predict[i],
                    "m%d_predict" % i: self._mom_decoder_predict[i]
                })
                if len(self._flows) > 0:
                    fetch_dict.update({
                        "flow%d" % i: self._flows[i]
                    })
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)


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
                 with_momentum=True,
                 rollout_depth=3,
                 rollout_lane=3,
                 max_rollout=5,
                 model_train_depth=3,
                 batch_size=32,
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
        :param args:
        :param kwargs:
        """

        def f_iaa(inputs):
            input_observation = inputs[0]
            input_action = inputs[1]
            action_dim = inputs[2]
            input_action = tf.one_hot(indices=input_action, depth=action_dim, on_value=1.0, off_value=0.0, axis=-1)

            net_se = network.Network([input_observation], f_se, var_scope="se_1")
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
        self._max_rollout = max_rollout
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
                depth=self._max_rollout,
                state_shape=state_shape,
                dim_action=num_action,
                transition_weight=1.0,
                with_momentum=with_momentum),
            name="env_model")
        # network_optimizer.freeze(self.network.sub_net("transition").variables)
        network_optimizer.compile()

        self._policy = StochasticPolicy(self._pi_distribution)

    def init_network(self, f_iaa, state_shape, num_action, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
        return network.Network([input_state, input_action, num_action], f_iaa, var_scope="learn")

    def update_on_trajectory(self, batch):
        if (np.shape(batch["action"])[0] >= self._max_rollout):
            # self.network_optimizer.update("policy_net", self.sess, batch)
            self.network_optimizer.update("env_model", self.sess, batch)
            self.network_optimizer.update("ac", self.sess, batch)
            self.network_optimizer.update("l2", self.sess)
            info = self.network_optimizer.optimize_step(self.sess)
            prefix = "EnvModelUpdater/env_model/"
            num = info[prefix + "num"]
            logging.warning("-----------%s steps for loss------------", num)
            if prefix+"s0" in info:
                s0 = info[prefix + "s0"]
                update_step = info[prefix + "update_step"]
                path_prefix = os.sep.join([self._log_dir, "Img", ""])
                if not os.path.isdir(path_prefix):
                    os.makedirs(path_prefix)
                logging.warning("writing images to %s", path_prefix)
                for i in range(len(s0)):
                    s = s0[i]
                    frame_n = s.shape[-1] / 3
                    for j in range(frame_n):
                        f = s[:, :,  3 * j: 3 * j + 3]
                        cv2.imwrite(path_prefix + "%d_%03d_f0_%d.png" % (update_step, i, j),
                                    cv2.cvtColor(255 * f.astype(np.float32), cv2.COLOR_RGB2BGR))
                    for d in range(num):
                        fn = info[prefix + "f%d" % d][i]
                        fn_predict = info[prefix + "f%d_predict" % d][i]
                        an_predict = info[prefix + "a%d_predict" % d][i]
                        mn_predict = info[prefix + "m%d_predict" % d][i]
                        flow = None
                        if prefix + "flow0" in info:
                            flow = info[prefix + "flow%d" % d][i]
                        # logging.warning("---------------------------------")
                        # logging.warning(np.mean(fn_predict))
                        # logging.warning(np.mean(an_predict))
                        # logging.warning(np.mean(mn_predict))

                        cv2.imwrite(path_prefix + "%d_%03d_f%d_raw.png" % (update_step, i, d+1),
                                    cv2.cvtColor(255 * fn.astype(np.float32), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(path_prefix + "%d_%03d_f%d_predict.png" % (update_step, i, d+1),
                                    cv2.cvtColor(255 * fn_predict.astype(np.float32), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(path_prefix + "%d_%03d_y_actionf%d_predict.png" % (update_step, i, d + 1),
                                    cv2.cvtColor(255 * an_predict.astype(np.float32), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(path_prefix + "%d_%03d_z_momf%d_predict.png" % (update_step, i, d + 1),
                                    cv2.cvtColor(255 * mn_predict.astype(np.float32), cv2.COLOR_RGB2BGR))
                        if flow is not None:
                            cv2.imwrite(path_prefix + "%d_%03d_g_flow%d.png" % (update_step, i, d + 1),
                                        flow_to_color(flow.astype(np.float32), max_len=16.0) * 255)
                del info[prefix + "s0"]
                del info[prefix + "update_step"]
                for d in range(self._max_rollout):
                    del info[prefix + "f%d" % d], info[prefix + "f%d_predict" % d]
                    del info[prefix + "a%d_predict" % d]
                    del info[prefix + "m%d_predict" % d]
                    if prefix + "flow0" in info:
                        del info[prefix + "flow%d" % d]
            return info, {}
        else:
            return {}, {}

    def set_session(self, sess):
        super(ActorCriticWithI2A, self).set_session(sess)
        self.network.set_session(sess)
        self._pi_distribution.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)

