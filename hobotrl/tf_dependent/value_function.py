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

# TODO: tf get_collection(key, scope) not working, check why
def get_collection(key, scope):
    """Temporary replacement for tf.get_collection()
    """
    vars = filter(
        lambda var: scope.name in var.name,
        tf.get_collection(key=key),
    )
    return vars

class DeepQFuncActionOut(object):
    """NN-Parameterized Action-Out Q Functions
    
    
    
    Build a Deep Q-Network (DQN) and Related TF Ops
    
    Graph:
    The Deep Q-Network and the target network.
    
    Training Ops:
    temporal difference
    target sync
    
    Public Value Methods:
    get_q_values:
    
    Public Symbol Methods:
    # graph, session, variable scope, train, fetch    
    """
    
    def __init__(self, gamma,
                 f_net, state_shape, num_actions, training_params, schedule,
                 batch_size,
                 greedy_policy=True, ddqn=False, graph=None, **kwargs):
        """Initialization
        Unpacks parameters, build related ops on graph (placeholders, networks,
        and training ops), and register handels to important ops.
        
        Note regularization losses and moving average ops are retrieved with
        variable scope and default graphkey with `tf.get_collection()`.
        Therefore f_net should properly register these ops to corresponding
        variable collections.

        Parameters
        ----------
        :param gamma : value discount factor.
        :param f_net : functional interface for building parameterized value fcn.
        :param state_shape : shape of state and next_state
        :param num_actions : number of actions
        :param training_params : parameters for training value fcn.. A tuple of
            two members:
                optimizer_td : Tensorflow optimizer for gradient-based opt.
                target_sync_rate: rate for copying the weights of the
                    non-target network to the target network. 1.0 means
                    hard-copy and [0, 1.0) means soft copy.
        :param schedule : periods of TD and Target Sync. ops. A length-2 tuple:
                n_step_td : steps between TD updates.
                n_step_sync : steps between weight syncs.
        :param greedy_policy : if evaluate the greedy policy.
        :param ddqn : True to enable Double DQN.
        :param graph : tf.Graph to build ops. Use default graph if None:
        """
        # Unpack params
        self.__GAMMA = gamma

        self.__F_NET = f_net
        self.__STATE_SHAPE = state_shape
        self.__NUM_ACTIONS = num_actions
        optimizer_td, target_sync_rate = training_params
        self.__optimizer_td = optimizer_td
        self.__TARGET_SYNC_RATE = target_sync_rate        
        self.__N_STEP_TD = schedule[0]
        self.__N_STEP_SYNC = schedule[1]
        self.countdown_td_ = self.__N_STEP_TD
        self.countdown_sync_ = self.__N_STEP_SYNC

        self.__GREEDY_POLICY = greedy_policy

        # Graph check
        if graph is None:
            graph = tf.get_default_graph()

        # Build ops
        with graph.as_default():
            with tf.variable_scope('DQN') as scope_dqn:
                # Placeholders
                state, action, reward, next_state, next_action, \
                importance, episode_done, is_training = \
                    self.__init_placeholders()

                # Intermediates
                with tf.variable_scope('non-target') as scope_non:
                    q = f_net(state, num_actions, is_training)
                    q_sel = tf.reduce_sum(
                        q * tf.one_hot(action, num_actions),
                        axis=1,
                        name='q_sel'
                    )
                # target network
                with tf.variable_scope('target') as scope_target:
                    next_q = f_net(next_state, num_actions, is_training)
                if greedy_policy:
                    # double dqn
                    if ddqn:
                        with tf.variable_scope('ddqn') as scope_double_q:
                            double_q = f_net(next_state, num_actions, is_training)

                        max_action = tf.argmax(double_q, axis=1)
                        max_action = tf.one_hot(max_action, num_actions, dtype=tf.float32)
                        next_q_sel = tf.reduce_sum(next_q * max_action, axis=1, keep_dims=True)
                    else:
                        next_q_sel = tf.reduce_max(next_q, axis=1, name='next_q_sel')
                else:
                    next_q_sel = tf.reduce_sum(
                        next_q * tf.one_hot(next_action, num_actions),
                        axis=1,
                        name='next_q_sel'
                    )

                target_q = tf.add(reward, gamma * next_q_sel * (1 - episode_done), name='target_q')
                td = importance * tf.subtract(
                     target_q,
                     q_sel,
                     name='td')
                td_loss = tf.reduce_mean(tf.square(td), name='td_loss')
                
                list_reg_loss = get_collection(
                    key=tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope_non
                )
                reg_loss = sum(list_reg_loss)

                non_target_vars = get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_non
                )
                target_vars = get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_target
                )
                # Training ops
                # td, TODO: merge moving_averages_op with op_train_td
                op_train_td = optimizer_td.minimize(
                    tf.add(td_loss, reg_loss, name='op_train_td'), var_list=non_target_vars
                )
                # sync
                op_list_sync_target = [
                    target_var.assign_sub(
                        target_sync_rate*(target_var-non_target_var)
                    ) for target_var, non_target_var in zip(target_vars, non_target_vars)
                ]
                target_diff_l2 = sum([
                    tf.reduce_sum(tf.square(target_var - non_target_var))
                    for target_var, non_target_var in zip(target_vars, non_target_vars)]
                )
                op_sync_target = tf.group(
                    *op_list_sync_target,
                    name='op_sync_target'
                )
                if ddqn:
                    ddqn_vars = get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_double_q)
                    op_sync_double_q = tf.group([tf.assign(var_d, var)
                                                 for var_d, var in zip(ddqn_vars, non_target_vars)])
                    with tf.control_dependencies([op_train_td]):
                        op_train_td = op_sync_double_q

        # Register op handles
        self.sym_state = state
        self.sym_action = action
        self.sym_reward = reward
        self.sym_next_state = next_state
        self.sym_next_action = next_action
        self.sym_importance = importance
        self.sym_episode_done = episode_done
        self.sym_is_training = is_training
        self.sym_q = q
        self.sym_q_sel = q_sel
        self.sym_next_q = next_q
        self.sym_next_q_sel = next_q_sel
        self.sym_td_loss = td_loss
        self.sym_target_diff_l2 = target_diff_l2
        self.op_train_td = op_train_td
        self.op_sync_target = op_sync_target
        
    def apply_op_sync_target_(self, sess, **kwargs):
        return sess.run(self.op_sync_target)

    def apply_op_train_td_(self, sess,
                       state, action, reward,
                       next_state, next_action=None,
                       episode_done=None, importance=None,
                       **kwargs):
        """Functional wrapper for fetching op_train_td
        Squeeze out redundant dims in action, reward, next_action,
        importance, and episode_done to match the shape of
        corresponding placeholders.
        
        Parameters
        ----------
        """
        feed_dict = {
            self.sym_state: state,
            self.sym_action: np.squeeze(action),
            self.sym_reward: np.squeeze(reward),
            self.sym_next_state: next_state
        }
        if self.__GREEDY_POLICY:
            next_action = None
            importance = None
        if next_action is not None:
            feed_dict[self.sym_next_action] = np.squeeze(next_action)
        if importance is not None:
            feed_dict[self.sym_importance] = np.squeeze(importance)
        if episode_done is not None:
            feed_dict[self.sym_episode_done] = np.squeeze(episode_done)

        return sess.run(self.op_train_td, feed_dict)

    def fetch_td_loss_(self, sess,
                       state, action, reward,
                       next_state, next_action=None,
                       episode_done=None, importance=None,
                       **kwargs):
        """Functional wrapper for fetching td_loss
        Squeeze out redundant dims in action, reward, next_action,
        importance, and episode_done to match the shape of
        corresponding placeholders.
        
        Parameters
        ----------
        """
        feed_dict = {
            self.sym_state: state,
            self.sym_action: np.squeeze(action),
            self.sym_reward: np.squeeze(reward),
            self.sym_next_state: next_state
        }
        if self.__GREEDY_POLICY:
            next_action = None
            importance = None
        if next_action is not None:
            feed_dict[self.sym_next_action] = np.squeeze(next_action)
        if importance is not None:
            feed_dict[self.sym_importance] = np.squeeze(importance)
        if episode_done is not None:
            feed_dict[self.sym_episode_done] = np.squeeze(episode_done)

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
        if sess is None:
            raise ValueError(
                "DeepQFuncActionOut.improve_value_(): "
                "please pass in a tf.Session()"
            )

        self.countdown_td_ -= 1
        self.countdown_sync_ -= 1
        info = {}
        td_loss = 0
        if self.countdown_td_ == 0:
            self.apply_op_train_td_(
                sess=sess, state=state, action=action,
                reward=reward, next_state=next_state, next_action=next_action,
                episode_done=episode_done, importance=importance,
                **kwargs
            )
            self.countdown_td_ = self.__N_STEP_TD
            td_loss = self.fetch_td_loss_(
                sess=sess, state=state, action=action,
                reward=reward, next_state=next_state, next_action=next_action,
                episode_done=episode_done, importance=importance,
                **kwargs
            )
            info["td_loss"] = td_loss
        
        if self.countdown_sync_ == 0:
            self.apply_op_sync_target_(sess=sess)
            self.countdown_sync_ = self.__N_STEP_SYNC
        
        return info

    def get_value(self, state, action=None, sess=None, **kwargs):
        if sess is None:
            raise ValueError(
                "DeepQFuncActionOut.get_value(): "
                "please pass in a tf.Session()"
            )
        action = np.squeeze(action) if action is not None else None

        if action is None:
            return sess.run(
                self.sym_q,
                feed_dict={self.sym_state: state}
            )
        else:
            return sess.run(
                self.sym_q_sel,
                feed_dict={
                    self.sym_state: state,
                    self.sym_action: action
                }
            )

    def get_subgraph_value(self):
        input_dict = {
            'state': self.sym_state,
            'action': self.sym_action,
            'is_training': self.sym_is_training
        }
        output_dict = {
            'q_all': self.sym_q,
            'q_sel': self.sym_q_sel
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
            'q_sel': self.sym_next_q_sel
        }
        return input_dict, output_dict

    def __init_placeholders(self):
        """Define Placeholders
        """
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
        ph = (
            state, action, reward, next_state, next_action,
            importance, episode_done, is_training
        )
        return ph
