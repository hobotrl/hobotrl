# -*- coding: utf-8 -*-

import logging

import tensorflow as tf
import numpy as np

import hobotrl as hrl
from distribution import DiscreteDistribution, NormalDistribution


class DeepStochasticPolicy(object):

    def __init__(self, state_shape, num_actions, is_continuous_action,
                 f_create_net, training_params, entropy, **kwargs):
        """

        :param state_shape:
        :param num_actions:
        :param is_continuous_action: True if output action of this policy is continuous
        :param f_create_net:
        :param training_params: tuple containing training parameters. One member:
                optimizer_spg : Tensorflow optimizer for gradient-based opt.
        :param entropy: entropy regularization term
        :param kwargs:
        """
        super(DeepStochasticPolicy, self).__init__(**kwargs)
        self.state_shape, self.num_actions, self.is_continuous_action = \
                state_shape, num_actions, is_continuous_action
        self.entropy = entropy
        self.optimizer_spg = training_params[0]
        # === initilize placeholders ===
        self.input_state = tf.placeholder(
            dtype=tf.float32, shape=[None] + list(state_shape), name="input_state"
        )
        if is_continuous_action:
            action_shape = [num_actions]
            self.input_action = tf.placeholder(
                dtype=tf.float32, shape=[None] + list(action_shape), name="input_action"
            )
        else:
            action_shape = []
            self.input_action = tf.placeholder(
                dtype=tf.int32, shape=[None], name="input_action"
            )
        self.action_shape = action_shape
        self.input_advantage = tf.placeholder(
            dtype=tf.float32, shape=[None], name="input_advantage"
        )
        self.input_entropy = tf.placeholder(
            dtype=tf.float32, name="input_entropy"
        )
        # === build net ===
        with tf.variable_scope("policy") as vs:
            if not is_continuous_action:
                self.distribution = DiscreteDistribution(
                    f_create_net, [self.input_state], num_actions,
                    input_sample=self.input_action, **kwargs
                )
            else:
                self.distribution = NormalDistribution(
                    f_create_net, [self.input_state], num_actions,
                    input_sample=self.input_action, **kwargs
                )
        vars_policy = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name
        )
        # === build operations === 
        self.op_entropy = tf.reduce_mean(self.distribution.entropy())
        self.spg_loss = -1.0 * tf.reduce_mean(
            self.distribution.log_prob() * self.input_advantage
        )
        self.reg_loss = sum(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope=vs.name
        )) + -1.0 * self.input_entropy * self.op_entropy
        self.total_loss = self.spg_loss + self.reg_loss
        self.op_train = self.optimizer_spg.minimize(
            self.total_loss, var_list=vars_policy
        )

    def act(self, state, sess, **kwargs):
        return self.distribution.sample_run(sess, [state])

    def improve_policy_(self, state, action, advantage, **kwargs):
        """Wrapper for training ops."""
        assert 'sess' in kwargs
        sess = kwargs['sess']

        _, loss, entropy = sess.run(
            [self.op_train, self.spg_loss, self.op_entropy],
            feed_dict={
                self.input_state: state,
                self.input_action: action,
                self.input_entropy: self.entropy,
                self.input_advantage: advantage
            }
        )

        return {
            "spg_loss": loss,
            "policy_entropy": entropy,
            "policy_adv": advantage,
        }


class DeepDeterministicPolicy(object):
    """Deterministic policy parameterized by a deep neural network.
    A deep deterministic policy (DDP) is a parameterized policy using deep
    neural networks. It outputs a deterministic action for each state. DDPs
    can be trained with the deterministic policy gradient (DPG):
        dpg = d Q(s, a)/ da * Hessian(a=pi(s), theta_pi).

    This class Build a DDP and related ops. Subgraph contains the non-target
    and targetpolicy network. Both network can output a action tensor given a
    state tensor. The non-target network is trained periodically with DPG.
    For decoupling purpose we do not use a symbolic reference for Q when
    computing DPG. Instead, the caller pass in the values of dQ/da, and we
    use it to reconstruct a symbolic local-linear expansion of Q:
        Q(s, a) ~= Q(S, a0) + dQ/da(s, a0) * (a - a0),
                = const. + dQ/da(s, a0)*(a - a0).
    The target network copies (soft or hard) the weights of the non-target
    network:
        theta_t = theta_t * (1-alpha) + theta_non * alpha.
    """
    def __init__(self, f_net, state_shape, action_shape,
                 training_params, schedule, graph=None,
                 **kwargs):
        """Initialize the deep deterministic policy.

        :param f_net: functional interface for building parameterized policy.
        :param state_shape: shape of each state sample.
        :param action_shape: shape of each action sample.
        :param training_params: policy training related parameters. Passed in
            as a tuple of 3 members. By position, these members are:
                0. optimizer_dpg: Tensorflow optimizer for DPG procedure.
                1. target_sync_rate: rate for copying the weights of the
                    non-target network to the target network. 1.0 means
                    hard-copy and [0, 1.0) means soft copy.
                2. max_dpg_grad_norm: maximum l2 norm for DPG.
        :param schedule: periods between calls for training ops. Passed in as
            a tuple of 2 members. By position, these members are:
                0. n_step_dpg : steps between DPG updates.
                1. n_step_sync: steps between weight synchronizations.
        :parm graph: the Tensorflow graph to build ops on. Use default graph
            if missing or None.
        """
        # === Unpack params ===
        # graph related
        self.__F_NET = f_net
        self.__STATE_SHAPE = state_shape
        self.__ACTION_SHAPE = action_shape
        if graph is None:
            graph = tf.get_default_graph()
        # optimization related
        optimizer_dpg = training_params[0]
        target_sync_rate = training_params[1]
        max_dpg_grad_norm = training_params[2]
        self.__optimizer_dpg = optimizer_dpg
        self.__TARGET_SYNC_RATE = target_sync_rate
        self.__MAX_DPG_GRAD_NROM = max_dpg_grad_norm
        # meta schedule
        self.__N_STEP_DPG = schedule[0]
        self.__N_STEP_SYNC = schedule[1]
        self.countdown_dpg_ = self.__N_STEP_DPG
        self.countdown_sync_ = self.__N_STEP_SYNC


        # === Build ops ===
        with graph.as_default():
            with tf.variable_scope('DDP') as scope_ddp:
                # placeholders
                state, next_state, grad_q_action, is_training = \
                        self.__init_placeholders()
                # policy network
                with tf.variable_scope('non-target') as scope_non:
                    action = f_net(state, action_shape, is_training)
                with tf.variable_scope('target') as scope_target:
                    next_action = f_net(next_state, action_shape, is_training)
                n_vars = tf.get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_non.name
                )
                t_vars = tf.get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_target.name
                )
                # dpg loss via local-linear expansion
                delta_q = tf.reduce_sum(
                    grad_q_action*(action - tf.stop_gradient(action)),
                    axis=1, name='delta_q'
                )
                dpg_loss = tf.reduce_mean(-1.0*delta_q, name='dpg_loss')
                # regularization loss
                list_reg_loss = tf.get_collection(
                    key=tf.GraphKeys.REGULARIZATION_LOSSES,
                    scope=scope_non.name
                )
                reg_loss = tf.add_n(list_reg_loss, name='reg_loss')
                # dpg, with regularization, grad clipping, and updates
                total_loss = tf.add(dpg_loss, reg_loss, name='total_loss')
                grad_and_vars = optimizer_dpg.compute_gradients(
                    total_loss, n_vars
                )
                list_clipped, dpg_grad_norm = tf.clip_by_global_norm(
                    [grad for grad, var in grad_and_vars], max_dpg_grad_norm
                )
                op_train_dpg = optimizer_dpg.apply_gradients(
                    [(grad_clipped, var) for grad_clipped, (grad, var) in
                     zip(list_clipped, grad_and_vars)]
                )
                ops_update = tf.get_collection(
                    key=tf.GraphKeys.UPDATE_OPS, scope=scope_non.name
                )
                op_train_dpg = tf.group(
                    op_train_dpg, *ops_update, name='op_train_td'
                )
                # sync
                op_list_sync_target = [
                    t_var.assign_sub(target_sync_rate*(t_var-n_var))
                    for t_var, n_var in zip(t_vars, n_vars)
                ]
                target_diff_l2 =tf.global_norm(
                  [t_var - n_var for t_var, n_var in zip(t_vars, n_vars)],
                  name='target_diff_l2'
                )
                op_sync_target = tf.group(
                    *op_list_sync_target, name='op_sync_target'
                )
                # force non-target and target network to take identical weights
                op_copy = [t_var.assign(n_var) for t_var, n_var in zip(t_vars, n_vars)]

        # === Register symbols ===
        # inputs
        self.sym_state = state
        self.sym_next_state = next_state
        self.sym_grad_q_action = grad_q_action
        self.sym_is_training = is_training
        # outputs and intermediate
        self.sym_action = action
        self.sym_next_action = next_action
        self.sym_delta_q = delta_q
        self.sym_dpg_loss = dpg_loss
        self.sym_dpg_grad_norm = dpg_grad_norm
        self.sym_target_diff_l2 = target_diff_l2
        # training ops
        self.op_train_dpg = op_train_dpg
        self.op_sync_target = op_sync_target
        self.op_copy = op_copy

    def apply_op_sync_target_(self, sess, **kwargs):
        """Apply op_sync_target."""
        return sess.run([self.op_sync_target, self.sym_target_diff_l2])

    def apply_op_train_dpg_(self, state, grad_q_action, sess=None, **kwargs):
        """Apply op_train_dpg."""
        feed_dict = {
            self.sym_state: state,
            self.sym_grad_q_action: grad_q_action
        }
        return sess.run([self.op_train_dpg, self.sym_dpg_grad_norm], feed_dict)

    def fetch_dpg_loss_(self, state, grad_q_action, sess=None, **kwargs):
        """Fetch dpg_loss."""
        feed_dict = {
            self.sym_state: state,
            self.sym_grad_q_action: grad_q_action
        }
        return sess.run(self.sym_dpg_loss, feed_dict)

    def improve_policy_(self, state, grad_q_action, sess=None, **kwargs):
        """Train non-target policy and sync target policy.
        The meta training procedure deep deterministic policy. Apply
        `op_train_dpg` and `op_sync_target` periodically with schedule specified by
        `self.__N_STEP_DPG` and `self.__N_STEP_SYNC`.

        :param state: a batch of state
        :param grad_q_action : a batch of value gradients w.r.t to action inputs.
        :param sess: tf session
        :param kwargs:
        :return: update info dict.
        """
        self.countdown_dpg_ -= 1 if self.countdown_dpg_>=0 else 0
        self.countdown_sync_ -= 1 if self.countdown_sync_>=0 else 0
        info = {}
        if self.countdown_dpg_ == 0:
            _, dpg_grad_norm = self.apply_op_train_dpg_(
                state=state, grad_q_action=grad_q_action,
                sess=sess, **kwargs
            )
            info.update({'dpg_grad_norm': dpg_grad_norm})
            self.countdown_dpg_ = self.__N_STEP_DPG
        if self.countdown_sync_ == 0:
            _, diff_l2 = self.apply_op_sync_target_(sess=sess)
            info.update({'ddp_target_diff_l2': diff_l2})
            self.countdown_sync_ = self.__N_STEP_SYNC
        return info

    def act(self, state, sess=None, is_training=True, use_target=False, **kwargs):
        """Fetch action for the provided batch of state."""
        if use_target:
            return sess.run(
                self.sym_next_action, feed_dict={
                    self.sym_next_state: state,
                    self.sym_is_training: is_training
                }
            )
        else:
            return sess.run(
                self.sym_action, feed_dict={
                    self.sym_state: state,
                    self.sym_is_training: is_training
                }
            )

    @property
    def state_shape(self):
        return self.__STATE_SHAPE

    @property
    def action_shape(self):
        return self.__ACTION_SHAPE

    def get_subgraph_policy(self):
        input_dict = {'state': self.sym_state}
        output_dict = {'action': self.sym_action}

        return input_dict, output_dict

    def __init_placeholders(self):
        """Define Placeholders."""
        state_shape = list(self.__STATE_SHAPE)
        action_shape = list(self.__ACTION_SHAPE)
        with tf.variable_scope('placeholders'):
            state = tf.placeholder(
                dtype=tf.float32, shape=[None]+state_shape, name='state'
            )
            next_state = tf.placeholder(
                dtype=tf.float32, shape=[None]+state_shape, name='next_state'
            )
            grad_q_action = tf.placeholder(
                dtype=tf.float32, shape=[None]+action_shape, name='grad_q_action'
            )
            is_training = tf.placeholder_with_default(
                input=True, shape=(), name='is_training'
            )
        return (state, next_state, grad_q_action, is_training)


