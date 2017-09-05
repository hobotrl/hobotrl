# -*- coding: utf-8 -*-
"""Value Function Parameterized with Deep NN
This module contains value function classes that are that are
parameterized by deep neural networks.

Each class encloses the related methods for building, training,
and looking up a particular type of value function. There are four
ways to access the functionalities of a value function instance.
    1. Training and looking up ops are exposed through public
    methods `improve_value_()` and `get_value()` for convenience.
    2. The corresponding subgraph is also accessable through the
    `get_subgraph_*()` methouds.
    3. The class will register the handle of important Tensorflow
    ops as public attributes for ad-hoc retrieval.
    4. These ops are also initialized with hierachical names to
    fascilitate retrieva by name from the tf.Graph passed in.
"""

import numpy as np

import tensorflow as tf
from hobotrl.utils import Network


class DeepQFuncActionOut(object):
    """NN-Parameterized Action-Out Q Functions
    Build the Deep Q-Network (DQN) in action-out form and set up related op.

    Subgraph includes the the non-target and target deep q-networks. Non-target
    network weights are trained with one-step temporal-difference backup:
            q(s, a) = r + gamma * q(s', a').
    And the target network weights are copied (soft or hard) from the non-target
    network following:
            theta_t = theta_t * (1-alpha) + theta_non * alpha.
    Both procedures are called periodically following a schedule defined by
    instance parameters.
    """

    def __init__(self, gamma, f_net, state_shape, num_actions,
                 training_params, schedule, greedy_policy=True, ddqn=False,
                 graph=None, **kwargs):
        """Initialization
        Unpacks parameters, build related ops on graph (placeholders, networks,
        and training ops), and register handels to important ops.

        Note regularization losses and moving average ops are retrieved with
        variable scope and default graphkey with `tf.get_collection()`.
        Therefore f_net should properly register these ops to corresponding
        variable collections.

        :param gamma: value discount factor.
        :param f_net: functional interface for building parameterized value fcn.
        :param state_shape: shape of state and next_state
        :param num_actions: number of actions
        :param training_params: parameters for training value fcn.. A tuple of
            two members:
                0. optimizer_td: Tensorflow optimizer for gradient-based opt.
                1. target_sync_rate: rate for copying the weights of the
                    non-target network to the target network. 1.0 means
                    hard-copy and [0, 1.0) means soft copy.
                2. max_td_grad_norm: maximum l2 norm for TD gradient.
        :param schedule: periods of TD and Target Sync. ops. A length-2 tuple:
                0. n_step_td: steps between TD updates.
                1. n_step_sync: steps between weight syncs.
        :param greedy_policy: if evaluate the greedy policy.
        :param ddqn: True to enable Double DQN.
        :param graph: tf.Graph to build ops. Use default graph if None
        """
        # === Unpack params ===
        # graph related
        self.__GAMMA = gamma
        self.__F_NET = f_net
        self.__STATE_SHAPE = state_shape
        self.__NUM_ACTIONS = num_actions
        if graph is None:
            graph = tf.get_default_graph()
        self.__GREEDY_POLICY = greedy_policy
        self.__DDQN = ddqn
        # optimization related
        optimizer_td = training_params[0]
        target_sync_rate = training_params[1]
        max_td_grad_norm = training_params[2]
        self.__optimizer_td = optimizer_td
        self.__TARGET_SYNC_RATE = target_sync_rate
        self.__MAX_TD_GRAD_NROM = max_td_grad_norm
        # meta schedule
        self.__N_STEP_TD = schedule[0]
        self.__N_STEP_SYNC = schedule[1]
        self.countdown_td_ = self.__N_STEP_TD
        self.countdown_sync_ = self.__N_STEP_SYNC

        # === Build ops ===
        with graph.as_default():
            with tf.variable_scope('DQN') as scope_dqn:
                # placeholders
                state, action, reward, next_state, next_action, \
                importance, episode_done, is_training = \
                    self.__init_placeholders()
                # value network
                with tf.variable_scope('non-target') as scope_non:
                    q = f_net(state, num_actions, is_training)
                    scope_non.reuse_variables()  # reuse non-target weights
                    double_q = f_net(state, num_actions, is_training)
                # target network
                with tf.variable_scope('target') as scope_target:
                    next_q = f_net(next_state, num_actions, is_training)
                n_vars = tf.get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=scope_non.name
                )
                t_vars = tf.get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=scope_target.name
                )
                # select Q values and actions
                q_sel = tf.reduce_sum(
                    q * tf.one_hot(action, num_actions), axis=1, name='q_sel'
                )
                if greedy_policy:
                    if ddqn:  # use non-target network to select the greedy action.
                        greedy_action = tf.argmax(
                            double_q, axis=1, name='action_greedy'
                        )
                        next_q_sel = tf.stop_gradient(tf.reduce_sum(
                            next_q*tf.one_hot(greedy_action, num_actions), axis=1
                        ), name='next_q_sel')
                    else:  # use target network to select the greedy actions
                        next_q_sel = tf.reduce_max(next_q, axis=1, name='next_q_sel')
                else:
                    next_q_sel = tf.reduce_sum(
                        next_q * tf.one_hot(next_action, num_actions),
                        axis=1, name='next_q_sel'
                    )
                # td loss
                td_target = tf.add(
                    reward, gamma*next_q_sel*(1-episode_done), name='td_target'
                )
                td = importance * tf.subtract(td_target, q_sel, name='td')
                td_losses = tf.square(td, name='td_losses')
                td_loss = tf.reduce_mean(td_losses, name='td_loss')
                # regularization loss
                list_reg_loss = tf.get_collection(
                    key=tf.GraphKeys.REGULARIZATION_LOSSES,
                    scope=scope_non.name
                )
                reg_loss = sum(list_reg_loss)
                # td grad, with regularization, grad clipped, and updates
                total_loss = tf.add(td_loss, reg_loss, name='op_train_td')
                grad_and_vars = optimizer_td.compute_gradients(
                     total_loss, n_vars
                )
                list_clipped, td_grad_norm = tf.clip_by_global_norm(
                    [grad for grad, var in grad_and_vars], max_td_grad_norm
                )
                op_train_td = optimizer_td.apply_gradients(
                    [(grad_clipped, var) for grad_clipped, (grad, var) in
                     zip(list_clipped, grad_and_vars)]
                )
                ops_update = tf.get_collection(
                    key=tf.GraphKeys.UPDATE_OPS, scope=scope_non.name
                )
                # TODO: what does this do? and what's the advantage?
                # op_train_td, gradients = Network.minimize_and_clip(optimizer_td, objective=tf.add(td_loss, reg_loss),
                #                                                   var_list=non_target_vars)

                op_train_td = tf.group(
                    op_train_td, *ops_update, name='op_train_td'
                )
                # sync
                op_list_sync_target = [
                    t_var.assign_sub(target_sync_rate*(t_var-n_var))
                    for t_var, n_var in zip(t_vars, n_vars)
                ]
                target_diff_l2 = sum([
                    tf.reduce_sum(tf.square(t_var - n_var))
                    for t_var, n_var in zip(t_vars, n_vars)]
                )
                op_sync_target = tf.group(
                    *op_list_sync_target, name='op_sync_target'
                )
                op_copy = [t_var.assign(n_var) for t_var, n_var in zip(t_vars, n_vars)]

        # === Register symbols ===
        # inputs
        self.sym_state = state
        self.sym_action = action
        self.sym_reward = reward
        self.sym_next_state = next_state
        self.sym_next_action = next_action
        self.sym_importance = importance
        self.sym_episode_done = episode_done
        self.sym_is_training = is_training
        # outputs and intermediates
        self.sym_q = q
        self.sym_double_q = double_q
        self.sym_q_sel = q_sel
        self.sym_next_q = next_q
        self.sym_next_q_sel = next_q_sel
        self.sym_td_target = td_target
        self.sym_td_loss = td_loss
        self.sym_td_losses = td_losses
        self.sym_td_grad_norm = td_grad_norm
        self.sym_regularization_loss = reg_loss
        self.sym_target_diff_l2 = target_diff_l2
        # training ops
        self.op_train_td = op_train_td
        self.op_sync_target = op_sync_target
        self.op_copy = op_copy
        self.sym_learnable_vars = n_vars 

    def apply_op_sync_target_(self, sess, **kwargs):
        """Apply op_sync_target"""
        return sess.run([self.op_sync_target, self.sym_target_diff_l2])

    def apply_op_train_td_(self, state, action, reward,
                           next_state, next_action=None,
                           episode_done=None, importance=None,
                           sess=None, **kwargs):
        """Apply op_train_td"""
        feed_dict = {
            self.sym_state: state,
            self.sym_action: action,
            self.sym_reward: reward,
            self.sym_next_state: next_state
        }
        if self.__GREEDY_POLICY:
            next_action = None
            importance = None
        if next_action is not None:
            feed_dict[self.sym_next_action] = next_action
        if importance is not None:
            feed_dict[self.sym_importance] = importance
        if episode_done is not None:
            feed_dict[self.sym_episode_done] = episode_done
        return sess.run([
            self.op_train_td, self.sym_td_loss, self.sym_td_losses,
            self.sym_q_sel, self.sym_next_q_sel, self.sym_td_target,
            self.sym_td_grad_norm], feed_dict=feed_dict
        )

    def fetch_td_loss_(self, state, action, reward,
                       next_state, next_action=None,
                       episode_done=None, importance=None,
                       sess=None, **kwargs):
        """Fetch td_loss"""
        feed_dict = {
            self.sym_state: state,
            self.sym_action: action,
            self.sym_reward: reward,
            self.sym_next_state: next_state
        }
        if self.__GREEDY_POLICY:
            next_action = None
            importance = None
        if next_action is not None:
            feed_dict[self.sym_next_action] = next_action
        if importance is not None:
            feed_dict[self.sym_importance] = importance
        if episode_done is not None:
            feed_dict[self.sym_episode_done] = episode_done

        return sess.run(self.sym_td_loss, feed_dict)

    def improve_value_(self,
                       state, action, reward,
                       next_state, next_action=None,
                       episode_done=None, importance=None,
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
        :param reward: a batch of reward
        :param next_state: a batch of next state
        :param next_action: a batch of next action
        :param episode_done: a batch of episode_done
        :param importance: a batch of importance, or scalar importance value
        :param sess: tf session
        :param kwargs:
        :return:
        """
        self.countdown_td_ -= 1 if self.countdown_td_>=0 else 0
        self.countdown_sync_ -= 1 if self.countdown_sync_>=0 else 0
        info = {}
        if self.countdown_td_ == 0:
            _, td_loss, td_losses, q, next_q, td_target, td_grad_norm = \
            self.apply_op_train_td_(
                sess=sess, state=state, action=action,
                reward=reward, next_state=next_state, next_action=next_action,
                episode_done=episode_done, importance=importance,
                **kwargs
            )
            self.countdown_td_ = self.__N_STEP_TD
            info.update({
                "td_loss": td_loss, "td_losses": td_losses,
                "actions": action,
                "rewards": reward,
                "done": episode_done,
                "q_vals": q, "next_q_vals": next_q,
                "td_target": td_target, 'td_grad_norm': td_grad_norm})
        if self.countdown_sync_ == 0:
            _, diff_l2 = self.apply_op_sync_target_(sess=sess)
            self.countdown_sync_ = self.__N_STEP_SYNC
            info.update({"dqn_target_diff_l2": diff_l2})
        return info

    def get_value(self, state, action=None, is_batch=True, **kwargs):
        """Retrieve value for a state (and action).
        Retrieve the action values of a given state or of a particular (state,
        action) tuple. Uses non-training mode if the sample passed in is a
        single example.

        :param state:
        :param action:
        :param is_batch: indicator for the batch case.
        """
        assert 'sess' in kwargs
        sess = kwargs['sess']
        if action is None:
            return sess.run(
                self.sym_q, feed_dict={
                    self.sym_state: state,
                    self.sym_is_training: is_batch
                }
            )
        else:
            return sess.run(
                self.sym_q_sel, feed_dict={
                    self.sym_state: state,
                    self.sym_action: action,
                    self.sym_is_training: is_batch
                }
            )

    def get_target_v(self, state, sess=None, **kwargs):
        """
        get state value from target network.
        :param state:
        :param sess:
        :param kwargs:
        :return:
        """
        return sess.run(
            self.sym_next_q_sel,
            feed_dict={self.sym_next_state: state}
        )

    @property
    def state_shape(self):
        return self.__STATE_SHAPE

    @property
    def greedy_policy(self):
        return self.__GREEDY_POLICY

    def get_subgraph_value(self):
        input_dict = {
            'state': self.sym_state,
            'action': self.sym_action,
            'is_training': self.sym_is_training
        }
        output_dict = {
            'q_all': self.sym_q,
        }
        return input_dict, output_dict

    def get_subgraph_value_target(self):
        input_dict = {
            'state': self.sym_next_state,
            'action': self.sym_next_action,
            'is_training': self.sym_is_training
        }
        output_dict = {
            'q_all': self.sym_next_q,
        }
        return input_dict, output_dict

    def __init_placeholders(self):
        """Define Placeholders"""
        state_shape = list(self.__STATE_SHAPE)
        with tf.variable_scope('placeholders'):
            state = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+state_shape,
                name='state'
            )
            action = tf.placeholder(
                dtype=tf.int32,
                shape=[None],
                name='action'
            )
            reward = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='reward'
            )
            next_state = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+state_shape,
                name='next_state'
            )
            next_action = tf.placeholder(
                dtype=tf.int32,
                shape=[None],
                name='next_action'
            )
            importance = tf.placeholder_with_default(
                tf.ones_like(action, dtype=tf.float32, name='default_importance'),
                shape=[None],
                name='importance'
            )
            episode_done = tf.placeholder_with_default(
                tf.zeros_like(action, dtype=tf.float32, name='default_episode_done'),
                shape=[None],
                name='episode_done'
            )
            is_training = tf.placeholder_with_default(
                input=True,
                shape=(),
                name='is_training'
            )
        placeholders = (
            state, action, reward, next_state, next_action,
            importance, episode_done, is_training
        )
        return placeholders


class DeepQFuncActionIn(object):
    """NN-Parameterized Action-In Q Functions
    Build the Deep Q-Network (DQN) in action-in form. Subgraphs include
    the the non-target and target deep q-networks.

    Non-target network weights are trained with one-step temporal-difference
    backup:
            q(s, a) = r + gamma * q(s', a').
    And target network weights are copied (soft or hard) from the non-target
    network:
            theta_t = theta_t * (1-alpha) + theta_non * alpha.
    Both procedures are called periodically following a schedule defined by
    instance parameters.
    """

    def __init__(self, gamma, f_net, state_shape, action_shape,
                 training_params, schedule, greedy_policy=True,
                 graph=None, **kwargs):
        """Initialization
        Unpacks parameters, build related ops on graph (placeholders, networks,
        and training ops), and register handels to important ops.

        Note regularization losses and moving average ops are retrieved with
        variable scope and default graphkey with `tf.get_collection()`.
        Therefore f_net should properly register these ops to corresponding
        variable collections.

        :param gamma: value discount factor.
        :param f_net: functional interface for building parameterized value fcn.
        :param state_shape: shape of state and next_state
        :param action_shape: shape of actions
        :param training_params: parameters for training value fcn.. A tuple of
            two members:
                0. optimizer_td: Tensorflow optimizer for gradient-based opt.
                1. target_sync_rate: rate for copying the weights of the
                    non-target network to the target network. 1.0 means
                    hard-copy and [0, 1.0) means soft copy.
                2. max_td_grad_norm: maximum l2 norm for TD gradient.
        :param schedule: periods of TD and Target Sync. ops. A length-2 tuple:
                0. n_step_td: steps between TD updates.
                1. n_step_sync: steps between weight syncs.
        :param greedy_policy: if evaluate the greedy policy.
        :param ddqn: True to enable Double DQN.
        :param graph: tf.Graph to build ops. Use default graph if None
        """
        # Unpack params
        # === Unpack params ===
        # graph related
        self.__GAMMA = gamma
        self.__F_NET = f_net
        self.__STATE_SHAPE = state_shape
        self.__ACTION_SHAPE = action_shape
        if graph is None:
            graph = tf.get_default_graph()
        self.__GREEDY_POLICY = greedy_policy
        # optimization related
        optimizer_td = training_params[0]
        target_sync_rate = training_params[1]
        max_td_grad_norm = training_params[2]
        self.__optimizer_td = optimizer_td
        self.__TARGET_SYNC_RATE = target_sync_rate
        self.__MAX_TD_GRAD_NROM = max_td_grad_norm
        # meta schedule
        self.__N_STEP_TD = schedule[0]
        self.__N_STEP_SYNC = schedule[1]
        self.countdown_td_ = self.__N_STEP_TD
        self.countdown_sync_ = self.__N_STEP_SYNC

        # === Build ops ===
        with graph.as_default():
            with tf.variable_scope('DQN') as scope_dqn:
                # placeholders
                state, action, reward, next_state, next_action, \
                importance, episode_done, is_training = \
                    self.__init_placeholders()
                # value network
                with tf.variable_scope('non-target') as scope_non:
                    q = f_net(state, action, is_training)
                with tf.variable_scope('target') as scope_target:
                    next_q = f_net(next_state, next_action, is_training)
                assert len(q.get_shape().dims)== 1
                assert len(next_q.get_shape().dims)== 1
                n_vars = tf.get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=scope_non.name
                )
                t_vars = tf.get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=scope_target.name
                )
                # get gradients:
                #   if q.shape = [batch_size,], action.shape = [batch_size]+action_shape,
                #   tf.gradients() will compute per-sample grad by default
                grad_q_action = tf.gradients(q, action, name='grad_q_action')[0]
                grad_q_action_t = tf.gradients(next_q, next_action, name='grad_q_action_t')[0]
                # td_loss
                td_target = tf.add(
                    reward, importance * gamma * next_q * (1 - episode_done),
                    name='target_q'
                )
                td = tf.subtract(td_target, q, name='td')
                td_losses = tf.square(td, name='td_losses')
                td_loss = tf.reduce_mean(td_losses, name='td_loss')
                # regularization loss
                list_reg_loss = tf.get_collection(
                    key=tf.GraphKeys.REGULARIZATION_LOSSES,
                    scope=scope_non.name
                )
                reg_loss = sum(list_reg_loss)
                # td grad
                total_loss = tf.add(td_loss, reg_loss, name='op_train_td')
                grad_and_vars = optimizer_td.compute_gradients(
                     total_loss, n_vars
                )
                list_clipped, td_grad_norm = tf.clip_by_global_norm(
                    [grad for grad, var in grad_and_vars], max_td_grad_norm
                )
                op_train_td = optimizer_td.apply_gradients(
                    [(grad_clipped, var) for grad_clipped, (grad, var) in
                     zip(list_clipped, grad_and_vars)]
                )
                ops_update = tf.get_collection(
                    key=tf.GraphKeys.UPDATE_OPS, scope=scope_non.name
                )
                op_train_td = tf.group(
                    op_train_td, *ops_update, name='op_train_td'
                )
                # sync
                op_list_sync_target = [
                    t_var.assign_sub(target_sync_rate*(t_var-n_var))
                    for t_var, n_var in zip(t_vars, n_vars)
                ]
                target_diff_l2 = sum([
                    tf.reduce_sum(tf.square(t_var - n_var))
                    for t_var, n_var in zip(t_vars, n_vars)]
                )
                op_sync_target = tf.group(
                    *op_list_sync_target, name='op_sync_target'
                )
                op_copy = [t_var.assign(n_var) for t_var, n_var in zip(t_vars, n_vars)]

        # === Register symbols ===
        # inputs
        self.sym_state = state
        self.sym_action = action
        self.sym_reward = reward
        self.sym_next_state = next_state
        self.sym_next_action = next_action
        self.sym_importance = importance
        self.sym_episode_done = episode_done
        self.sym_is_training = is_training
        # outputs and intermediates
        self.sym_q = q
        self.sym_next_q = next_q
        self.sym_grad_q_action = grad_q_action
        self.sym_grad_q_action_t = grad_q_action_t
        self.sym_td_grad_norm = td_grad_norm
        self.sym_td_target = td_target
        self.sym_td_loss = td_loss
        self.sym_td_losses = td_losses
        self.sym_target_diff_l2 = target_diff_l2
        self.op_train_td = op_train_td
        self.op_sync_target = op_sync_target
        self.op_copy = op_copy

    def apply_op_sync_target_(self, sess, **kwargs):
        """Apply op_sync_target."""
        return sess.run([self.op_sync_target, self.sym_target_diff_l2])

    def apply_op_train_td_(self, state, action, reward,
                           next_state, next_action,
                           episode_done=None, importance=None,
                           sess=None, **kwargs):
        """Apply op_train_td"""
        feed_dict = {
            self.sym_state: state,
            self.sym_action: action,
            self.sym_reward: reward,
            self.sym_next_state: next_state,
            self.sym_next_action: next_action
        }
        if importance is not None:
            feed_dict[self.sym_importance] = importance
        if episode_done is not None:
            feed_dict[self.sym_episode_done] = episode_done
        return sess.run([
            self.op_train_td, self.sym_td_loss, self.sym_td_losses,
            self.sym_q, self.sym_next_q, self.sym_td_target,
            self.sym_td_grad_norm], feed_dict=feed_dict
        )

    def fetch_td_loss_(self, state, action, reward,
                 next_state, next_action,
                 episode_done=None, importance=None,
                 sess=None, **kwargs):
        """Fetch td_loss"""
        feed_dict = {
            self.sym_state: state,
            self.sym_action: action,
            self.sym_reward: reward,
            self.sym_next_state: next_state,
            self.sym_next_action: np.squeeze(next_action)
        }
        if importance is not None:
            feed_dict[self.sym_importance] = importance
        if episode_done is not None:
            feed_dict[self.sym_episode_done] = episode_done
        return sess.run(self.sym_td_loss, feed_dict)

    def improve_value_(self, state, action, reward,
                 next_state, next_action,
                 episode_done=None, importance=None,
                 sess=None, **kwargs):
        """Interface for Training Value Fcn.
        The Deep Q-Network training procedure: apply `op_train_td`
        and `op_sync_target` with the periodic schedule specified by
        `self.__N_STEP_TD` and `self.__N_STEP_SYNC`.

        :param state: a batch of state
        :param action: a batch of action
        :param reward: a batch of reward
        :param next_state: a batch of next state
        :param next_action: a batch of next action
        :param episode_done: a batch of episode_done
        :param importance: a batch of importance, or scalar importance value
        :param sess: tf session
        :param kwargs:
        :return:
        """
        self.countdown_td_ -= 1 if self.countdown_td_>=0 else 0
        self.countdown_sync_ -= 1 if self.countdown_sync_>=0 else 0
        info = {}
        if self.countdown_td_ == 0:
            _, td_loss, td_losses, q, next_q, td_target, td_grad_norm = \
            self.apply_op_train_td_(
                sess=sess, state=state, action=action,
                reward=reward, next_state=next_state, next_action=next_action,
                episode_done=episode_done, importance=importance,
                **kwargs
            )
            self.countdown_td_ = self.__N_STEP_TD
            info.update({
                "td_loss": td_loss, "td_losses": td_losses,
                "q_vals": q, "next_q_vals": next_q,
                "td_target": td_target, 'td_grad_norm': td_grad_norm})
        if self.countdown_sync_ == 0:
            _, diff_l2 = self.apply_op_sync_target_(sess=sess)
            self.countdown_sync_ = self.__N_STEP_SYNC
            info.update({"dqn_target_diff_l2": diff_l2})
        return info

    def get_value(self, state, action, is_batch=True, **kwargs):
        """Retrieve value for a state (and action).
        Retrieve the action value of a given (state, action) tuple. Uses
        non-training mode if the sample passed in is a single example.

        :param state:
        :param action:
        :param is_batch: indicator for the batch case.
        """
        assert 'sess' in kwargs
        sess = kwargs['sess']
        return sess.run(
            self.sym_q, feed_dict={
                self.sym_state: state,
                self.sym_action: action,
                self.sym_is_training: is_batch
            }
        )

    def get_grad_q_action(self, state, action, sess=None,
                          use_target=False, **kwargs):
        if not use_target:
            feed_dict = {self.sym_state: state, self.sym_action: action}
            return sess.run(self.sym_grad_q_action, feed_dict)
        else:
            feed_dict = {self.sym_next_state: state, self.sym_next_action: action}
            return sess.run(self.sym_grad_q_action_t, feed_dict)

    @property
    def state_shape(self):
        """State shape getter
        """
        return self.__STATE_SHAPE

    @property
    def action_shape(self):
        """Action shape getter
        """
        return self.__ACTION_SHAPE

    @property
    def greedy_policy(self):
        return self.__GREEDY_POLICY

    def get_subgraph_value(self):
        input_dict = {
            'state': self.sym_state,
            'action': self.sym_action,
            'is_training': self.sym_is_training
        }
        output_dict = {
            'q': self.sym_q,
        }
        return input_dict, output_dict

    def get_subgraph_value_target(self):
        input_dict = {
            'state': self.sym_next_state,
            'action': self.sym_next_action,
            'is_training': self.sym_is_training
        }
        output_dict = {
            'q': self.sym_next_q,
        }
        return input_dict, output_dict

    def __init_placeholders(self):
        """Define Placeholders
        """
        state_shape = list(self.__STATE_SHAPE)
        action_shape = list(self.__ACTION_SHAPE)
        with tf.variable_scope('placeholders'):
            state = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+state_shape,
                name='state'
            )
            action = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+action_shape,
                name='action'
            )
            reward = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='reward'
            )
            next_state = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+state_shape,
                name='next_state'
            )
            next_action = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+action_shape,
                name='next_action'
            )
            importance = tf.placeholder_with_default(
                tf.ones_like(reward, dtype=tf.float32, name='default_importance'),
                shape=[None],
                name='importance'
            )
            episode_done = tf.placeholder_with_default(
                tf.zeros_like(reward, dtype=tf.float32, name='default_episode_done'),
                shape=[None],
                name='episode_done'
            )
            is_training = tf.placeholder_with_default(
                input=True,
                shape=(),
                name='is_training'
            )
        placeholders = (
            state, action, reward, next_state, next_action,
            importance, episode_done, is_training
        )
        return placeholders

