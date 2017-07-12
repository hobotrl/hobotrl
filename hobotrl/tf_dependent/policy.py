#
# -*- coding: utf-8 -*-

import logging

import tensorflow as tf
import numpy as np

import hobotrl as hrl
from distribution import *


class NNStochasticPolicy(object):

    def __init__(self, parent_agent, state_shape, num_actions, f_create_net, is_continuous_action=False,
                 training_params=None, entropy=0.01, gamma=0.9, train_interval=8, **kwargs):
        """

        :param parent_agent: agent which this policy belongs to.
                in order to get access to other modules in the same agent.
        :param state_shape:
        :param num_actions:
        :param f_create_net:
        :param is_continuous_action: True if output action of this policy is continuous
        :param training_params: tuple containing training parameters.
            two members:
                optimizer_td : Tensorflow optimizer for gradient-based opt.
                target_sync_rate: for other use.
        :param entropy: entropy regularization term
        :param gamma: reward discount factor
        :param train_interval: policy update interval
        :param kwargs:
        """
        kwargs.update({
            "state_shape": state_shape,
            "num_actions": num_actions,
            "training_params": training_params,
            "entropy": entropy,
            "gamma": gamma,
            "train_interval": train_interval
        })
        super(NNStochasticPolicy, self).__init__()
        self.parent_agent = parent_agent
        self.state_shape, self.num_actions = state_shape, num_actions
        self.entropy, self.reward_decay, self.train_interval = entropy, gamma, train_interval
        self.is_continuous_action = is_continuous_action
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="input_state")
        if is_continuous_action:
            action_shape = [num_actions]
            self.input_action = tf.placeholder(dtype=tf.float32, shape=[None] + action_shape, name="input_action")
        else:
            action_shape = []
            self.input_action = tf.placeholder(dtype=tf.int32, shape=[None], name="input_action")
        self.input_advantage = tf.placeholder(dtype=tf.float32, shape=[None], name="input_advantage")
        self.input_entropy = tf.placeholder(dtype=tf.float32, name="input_entropy")
        with tf.variable_scope("policy") as vs:
            if not is_continuous_action:
                self.distribution = DiscreteDistribution(f_create_net, [self.input_state], num_actions,
                                                         input_sample=self.input_action, **kwargs)
            else:
                self.distribution = NormalDistribution(f_create_net, [self.input_state], num_actions,
                                                       input_sample=self.input_action, **kwargs)
        vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
        self.episode_buffer = hrl.playback.MapPlayback(train_interval, {
            "s": state_shape,
            "a": action_shape,
            "r": [],
            "t": [],
            "s1": state_shape
        }, pop_policy="sequence")

        # other operators for training
        self.op_entropy = tf.reduce_mean(self.distribution.entropy())
        self.pi_loss = tf.reduce_mean(self.distribution.log_prob() * self.input_advantage) \
                       + self.input_entropy * self.op_entropy
        self.pi_loss = -self.pi_loss
        if training_params is None:
            optimizer = tf.train.AdamOptimizer()
        else:
            optimizer = training_params[0]

            # / mean
            # sigma
            # a normalize

        self.op_train = optimizer.minimize(self.pi_loss, var_list=vars_policy)
        # self.pi_loss = self.pi_loss + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), "policy")
        self.train_countdown = self.train_interval

    def act(self, state, sess, **kwargs):
        return self.distribution.sample_run(sess, [np.asarray([state])])

    def update_policy(self, state, action, reward, next_state, episode_done, sess, **kwargs):
        self.episode_buffer.push_sample(
            {
                's': state,
                'a': action,
                's1': next_state,
                'r': np.asarray([reward], dtype=float),
                't': np.asarray([episode_done], dtype=float)
            }, reward
        )
        self.train_countdown -= 1
        if episode_done or self.train_countdown == 0:
            self.train_countdown = self.train_interval
            batch_size = self.episode_buffer.get_count()
            batch = self.episode_buffer.sample_batch(batch_size)
            self.episode_buffer.reset()
            Si, Ai, Ri, Sj, T = np.asarray(batch['s'], dtype=float), batch['a'], batch['r'], \
                                np.asarray(batch['s1'], dtype=float), batch['t']
            # computing advantage
            R = np.zeros(shape=[batch_size], dtype=float)
            V = self.get_value(state=Si, sess=sess)
            if episode_done:
                r = 0.0
            else:
                last_v = self.get_value(state=np.asarray([next_state]), sess=sess)
                r = last_v[0]

            for i in range(batch_size):
                index = batch_size - i - 1
                if T[index] != 0:
                    # logging.warning("Terminated!, Ri:%s, Vi:%s", Ri[index], Vi[index])
                    r = 0
                r = Ri[index] + self.reward_decay * r
                R[index] = r
            advantage = R - V
            _, loss, entropy = sess.run([self.op_train, self.pi_loss, self.op_entropy],
                                        feed_dict={self.input_state: Si,
                                                   self.input_action: Ai,
                                                   self.input_entropy: self.entropy,
                                                   self.input_advantage: advantage})

            return {"policy_loss": loss, "entropy": entropy, "advantage": advantage, "V": V}

        return {}

    def get_value(self, state, sess):
        if self.is_continuous_action:
            action = self.distribution.mean_run(sess, [state])
            V = self.parent_agent.get_value(state=state, action=action)
        else:
            V = self.parent_agent.get_value(state=state)
            V = np.max(V, axis=-1, keepdims=False)
        return V


class DeepDeterministicPolicy(object):
    """Deterministic policy parameterized by a deep neural network.
    Build a deep deterministic policy and related op.

    Subgraph contains a policy network which outputs a deterministic action
    tensor given a state tensor. The policy network is trained periodically
    with the deterministic policy gradient (DPG):
        dpg = d Q(s, a)/ da * Hessian(a=pi(s), theta_pi).
    Q value is reconstructed via local linear expansion:
        Q(s0, a) ~= Q(S0, a0) + dQ/da * (a - a0),
                  = const. + dQ/da * a.
    """
    def __init__(self, f_net_ddp, state_shape, action_shape,
                 training_params, schedule, batch_size,
                 graph=None, **kwargs):
        """Initialization
        Parameters
        ----------
        f_net_ddp : functional interface for building parameterized policy fcn.
        state_shape : shape of state.
        action_shape : shape of action.
        training_params : parameters for training value fcn.. A tuple of one
            member:
                optimizer_dpg: Tensorflow optimizer for gradient-based opt.
        schedule : periods of dpg ops. A length-1 tuple:
                n_step_dpg : steps between DGP updates.
        batch_size : batch size
        graph : tf.Graph to build ops. Use default graph if None
        """
        # === Unpack params ===
        self.__F_NET = f_net_ddp
        self.__STATE_SHAPE = state_shape
        self.__ACTION_SHAPE = action_shape
        optimizer_dpg = training_params[0]
        self.__optimizer_dpg = optimizer_dpg
        self.__N_STEP_DPG = schedule[0]
        self.countdown_dpg_ = self.__N_STEP_DPG

        # === Graph check ===
        if graph is None:
            graph = tf.get_default_graph()

        # === Build ops ===
        with graph.as_default():
            with tf.variable_scope('DDP') as scope_ddp:
                # === Build Placeholders ===
                state, grad_q_action, is_training = \
                    self.__init_placeholders()

                # === Intermediates ===
                with tf.variable_scope('policy') as scope_pi:
                    action = f_net_ddp(state, action_shape, is_training)

                policy_vars = tf.get_collection(
                    key=tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=scope_pi.name
                )

                # dpg loss                
                dpg_loss = tf.reduce_sum(
                    -1.0*grad_q_action*action,
                    name='dpg_loss'
                )

                # regularization loss
                list_reg_loss = tf.get_collection(
                    key=tf.GraphKeys.REGULARIZATION_LOSSES,
                    scope=scope_pi.name
                )
                reg_loss = sum(list_reg_loss)

                # dpg, TODO: merge moving_averages_op with op_train_td
                op_train_dpg = optimizer_dpg.minimize(
                    tf.add(dpg_loss, reg_loss, name='op_train_dpg'),
                    var_list=policy_vars
                )

        self.sym_state = state
        self.sym_grad_q_action = grad_q_action
        self.sym_is_training = is_training
        self.sym_action = action
        self.sym_dpg_loss = dpg_loss
        self.op_train_dpg = op_train_dpg        

    def apply_op_train_dpg_(self, state, grad_q_action, sess=None, **kwargs):
        """Wrapper method for evaluating op_train_dpg"""
        feed_dict = {
            self.sym_state: state,
            self.sym_grad_q_action: grad_q_action
        }

        return sess.run([self.op_train_dpg, self.sym_dpg_loss], feed_dict)

    def fetch_dpg_loss_(self, state, grad_q_action, sess=None, **kwargs):
        """Wrapper method for fetching dpg_loss"""
        feed_dict = {
            self.sym_state: state,
            self.sym_grad_q_action: grad_q_action
        }

        return sess.run(self.sym_dpg_loss, feed_dict)

    def improve_policy_(self, state, grad_q_action, sess=None, **kwargs):
        """Interface for Training Policy Network
        The deep deterministic policy training procedure: apply `op_train_dpg`
        with the periodic schedule specified by `self.__N_STEP_DPG`.

        Parameters
        ----------
        :param state: a batch of state
        :param grad_q_action : a batch of value gradients w.r.t to action inputs.
        :param sess: tf session
        :param kwargs:
        :return:
        """
        self.countdown_dpg_ -= 1
        info = {}
        if self.countdown_dpg_ == 0:
            _, dpg_loss = self.apply_op_train_dpg_(
                state=state, grad_q_action=grad_q_action,
                sess=sess, **kwargs
            )
            info['dpg_loss'] = dpg_loss
            self.countdown_dpg_ = self.__N_STEP_DPG

        return info

    def act(self, state, sess=None, **kwargs):
        return sess.run(self.sym_action, {self.sym_state: state})

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
        """Define Placeholders"""
        state_shape = list(self.__STATE_SHAPE)
        action_shape = list(self.__ACTION_SHAPE)
        with tf.variable_scope('placeholders'):
            state = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+state_shape,
                name='state'
            )
            grad_q_action = tf.placeholder(
                dtype=tf.float32,
                shape=[None]+action_shape,
                name='grad_q_action'
            )
            is_training = tf.placeholder_with_default(
                input=True,
                shape=(),
                name='is_training'
            )
        return (state, grad_q_action, is_training)
    
