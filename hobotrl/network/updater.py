# -*- coding: utf-8 -*-

import tensorflow as tf
import network


class L2(network.NetworkUpdater):

    def __init__(self, net_or_var_scope):
        super(L2, self).__init__()
        if isinstance(net_or_var_scope, network.Network):
            var_scope = net_or_var_scope.abs_var_scope
        else:
            var_scope = net_or_var_scope
        self._var_scope = var_scope
        l2_loss = network.Utils.scope_vars(var_scope, tf.GraphKeys.REGULARIZATION_LOSSES)
        var_list = network.Utils.scope_vars(var_scope)
        self._l2_loss = tf.add_n(l2_loss)
        self._update_operation = network.MinimizeLoss(self._l2_loss, var_list=var_list)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, *args, **kwargs):
        return network.UpdateRun(fetch_dict={"l2_loss": self._l2_loss})


class FitTargetQ(network.NetworkUpdater):

    def __init__(self, learn_q, target_estimator):
        """
        perform 1-step td update to NetworkFunction learn_q according to NetworkFunction target_q
        :param learn_q:
        :type learn_q: network.NetworkFunction
        :param target_estimator:
        :type target_estimator: target_estimate.TargetEstimator
        """
        super(FitTargetQ, self).__init__()
        self._q, self._target_estimator = learn_q, target_estimator
        self._num_actions = learn_q.output().op.shape.as_list()[-1]
        self._input_target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_q")
        self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
        op_q = learn_q.output().op
        one_hot = tf.one_hot(self._input_action, self._num_actions)
        selected_q = tf.reduce_sum(one_hot * op_q, axis=1)
        self._op_losses = network.Utils.clipped_square(
                self._input_target_q - selected_q
            )
        self._sym_loss = tf.reduce_mean(
            self._op_losses
        )
        self._update_operation = network.MinimizeLoss(self._sym_loss, var_list=self._q.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        target_q_val = self._target_estimator.estimate(
            batch["state"], batch["action"], batch["reward"], batch["next_state"], batch["episode_done"])
        feed_dict = {self._input_target_q: target_q_val, self._input_action: batch["action"]}
        feed_dict.update(self._q.input_dict(batch["state"]))
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"target_q": target_q_val,
                                                                  "td_loss": self._sym_loss,
                                                                  "td_losses": self._op_losses})
