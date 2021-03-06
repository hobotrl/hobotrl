# -*- coding: utf-8 -*-

import tensorflow as tf
import logging
from hobotrl import sampling

from hobotrl.algorithms.iaa import EnvModelUpdater
from hobotrl.algorithms.ot import OTDQN, TrajectoryFitQ
import hobotrl.network as network
from hobotrl.algorithms.value_based import GreedyStateValueFunction
from hobotrl.playback import MapPlayback
from hobotrl.target_estimate import OptimalityTighteningEstimator


class OTModel(OTDQN):
    def __init__(self, f_create_q, f_se, f_transition, f_decoder,
                 # optimality tightening parameters
                 lower_weight, upper_weight, state_shape, num_actions,
                 # env model parameters
                 rollout_depth,
                 discount_factor,
                 ddqn, target_sync_interval, target_sync_rate, greedy_epsilon, network_optimizer=None,
                 max_gradient=10.0, update_interval=4, replay_size=1000, batch_size=32, sampler=None,
                 with_momentum = True,
                 curriculum=[1, 3, 5],
                 skip_step=[10000, 20000],
                 save_image_interval=10000,
                 log_dir=None,
                 with_ob=False,
                 with_goal=True,
                 *args, **kwargs):
        kwargs.update({
            "f_se": f_se,
            "f_transition": f_transition,
            "f_decoder": f_decoder,
            "rollout_depth": rollout_depth,
            "log_dir": log_dir
        })
        self._state_shape, self._num_actions = state_shape, num_actions
        self._rollout_depth = rollout_depth
        self._with_momentum = with_momentum
        self._curriculum, self._skip_step = curriculum, skip_step
        self._save_image_interval = save_image_interval
        self._with_ob = with_ob
        self._with_goal = with_goal
        if sampler is None:
            max_traj_length = 200
            sampler = sampling.TruncateTrajectorySampler2(None, replay_size / max_traj_length, max_traj_length,
                                                          batch_size=1, trajectory_length=batch_size, interval=update_interval)

        super(OTModel, self).__init__(f_create_q, lower_weight, upper_weight, batch_size, state_shape, num_actions,
                                      discount_factor, ddqn, target_sync_interval, target_sync_rate, greedy_epsilon,
                                      network_optimizer, max_gradient, update_interval, replay_size, batch_size,
                                      sampler, *args, **kwargs)
        self._log_dir = log_dir

    def init_network(self, f_create_q, f_se, f_transition, f_decoder, state_shape, num_actions, *args, **kwargs):
        def f(inputs):
            input_state, input_action, input_frame = inputs[0], inputs[1], inputs[2]
            action_onehot = tf.one_hot(indices=input_action, depth=num_actions, on_value=1.0, off_value=0.0, axis=-1)
            net_se = network.Network([input_state], f_se, var_scope="state_encoder")
            se = net_se["se"].op

            if not self._with_ob:
                net_transition = network.Network([se, action_onehot], f_transition, var_scope="TranModel")
                net_decoder = network.Network([tf.concat((se, se), axis=-1), input_frame], f_decoder, var_scope="Decoder")
            else:
                net_transition = network.Network([input_state, action_onehot], f_transition, var_scope="ObTranModel")
                net_decoder = network.Network([input_frame], f_decoder, var_scope="ObDecoder")
            net_q = network.Network([se], f_create_q, var_scope="q")
            return {
                "q": net_q["q"].op,
            }, {
                "se": net_se,
                "decoder": net_decoder,
                "transition": net_transition
            }

        input_frame = tf.placeholder(dtype=tf.float32, shape=[None, state_shape[0], state_shape[1], 3],
                                     name="input_frame")
        input_state = tf.placeholder(dtype=tf.float32, shape=[None]+list(state_shape), name="input_state")
        input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
        return network.NetworkWithTarget([input_state, input_action, input_frame], f, var_scope="learn", target_var_scope="target")

    def init_updaters_(self):
        self.target_v = GreedyStateValueFunction(self.target_q)
        target_esitmator = OptimalityTighteningEstimator(self.target_v, self._upper_weight, self._lower_weight,
                                                         discount_factor=self._discount_factor)

        self.network_optimizer.add_updater(TrajectoryFitQ(self.learn_q, target_esitmator), name="ot")
        self.network_optimizer.add_updater(network.L2(self.network), name="l2")
        self.network_optimizer.add_updater(EnvModelUpdater(
            self.network.sub_net("se"),
            self.network.sub_net("transition"),
            self.network.sub_net("decoder"),
            state_shape=self._state_shape,
            dim_action=self._num_actions,
            # curriculum=[1, self._rollout_depth],
            # skip_step=[10000],
            # transition_weight=1.0, with_momentum=True
            curriculum=self._curriculum,
            skip_step=self._skip_step,
            transition_weight=1.0,
            with_momentum=self._with_momentum,
            save_image_interval=self._save_image_interval,
            with_ob=self._with_ob,
            with_goal=self._with_goal
        ), name="env")
        self.network_optimizer.compile()

    def init_value_function(self, **kwargs):
        self.learn_q = network.NetworkFunction(self.network["q"], inputs=[self.network.inputs[0]])
        self.target_q = network.NetworkFunction(self.network.target["q"], inputs=[self.network.inputs[0]])
        self.target_v = GreedyStateValueFunction(self.target_q)
        return self.learn_q

    def update_on_transition(self, batch):
        if batch is None or len(batch) == 0:
            return {}, {}
        traj0 = batch[0]
        if len(traj0["action"]) < self._rollout_depth:
            return {}, {}
        self._update_count += 1
        self.network_optimizer.update("ot", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        self.network_optimizer.update("env", self.sess, traj0)

        info = self.network_optimizer.optimize_step(self.sess)
        EnvModelUpdater.check_save_image("env", info, self._log_dir)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {}

