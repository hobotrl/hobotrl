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
    """Fit Q network using temporal difference (TD) method.

    This class fits a symbolic Q network to estimated target values.
    """
    def __init__(self, learn_q, target_estimator, td_loss_fcn=None):
        """Initialization.

        Define the symbolic graph for calculating and minimizing loss.
        :param learn_q: symbolic Q function.
        :type learn_q: network.NetworkFunction
        :param target_estimator: estimator for target Q values.
        :type target_estimator: target_estimate.TargetEstimator
        :param td_loss_fcn: callable to map temporal difference to loss.
            Default value is tf.square.
        """
        super(FitTargetQ, self).__init__()
        # unpack params
        self._q, self._target_estimator = learn_q, target_estimator
        if td_loss_fcn is None:
            td_loss_fcn = tf.square
        # need computed target Q values and selected action as input
        self._input_target_q = tf.placeholder(
            dtype=tf.float32, shape=[None], name="input_target_q")
        self._input_action = tf.placeholder(
            dtype=tf.uint8, shape=[None], name="input_action")
        self._input_sample_weight = tf.placeholder_with_default([1.0], shape=[None], name="input_weight")
        op_q = learn_q.output().op
        num_actions = learn_q.output().op.shape.as_list()[-1]
        self.selected_q = tf.reduce_sum(
            tf.one_hot(self._input_action, num_actions) * op_q, axis=1)
        self._op_td = self.selected_q - self._input_target_q
        self._op_losses = td_loss_fcn(self._op_td)
        self._op_losses_weighted = self._op_losses * self._input_sample_weight
        self._sym_loss = tf.reduce_mean(self._op_losses_weighted)
        self._update_operation = network.MinimizeLoss(self._sym_loss, var_list=self._q.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        """Perform fitting on a batch of data.

        Assumes the batch passed in contains (s, a, r, s', done)
        transitions.

        :param sess: TensorFlow session where variables are stored.
        :param batch: data batch used to calculate loss and update netw.
        :return: UpdateRun instance with feed and fetch specs.
        """
        # Calculated target Q values using target estimator
        assert "state" in batch and "action" in batch and \
               "reward" in batch and "next_state" in batch and \
               "episode_done" in batch
        target_q_val = self._target_estimator.estimate(
            batch["state"], batch["action"], batch["reward"],
            batch["next_state"], batch["episode_done"])

        # Prepare data and fit Q network
        feed_dict = {self._input_target_q: target_q_val,
                     self._input_action: batch["action"]}
        if "_weight" in batch:
            feed_dict[self._input_sample_weight] = batch["_weight"]
        feed_dict.update(self._q.input_dict(batch["state"]))
        fetch_dict = {
            "action": batch["action"], "reward": batch["reward"],
            "done": batch["episode_done"],
            "q": self.selected_q, "target_q": target_q_val,
            "optimizer_loss": self._sym_loss,
            "td": self._op_td,
            "td_losses": self._op_losses,
            "td_losses_weighted": self._op_losses_weighted}
        update_run = network.UpdateRun(feed_dict=feed_dict, fetch_dict=fetch_dict)

        return update_run

