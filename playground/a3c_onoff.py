# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl
from hobotrl.utils import Network


def escape(name):
    return name.replace(":", "/")


class ActorCritic(object):

    def __init__(self, id, name, state_shape, num_actions, create_net, optimizer, parent=None, global_step=None,
                 ddqn=False, aux_r=False, aux_d=False, reward_decay=0.99, prob_min=1e-2):
        """

        :param id:
        :param name:
        :param state_shape:
        :param num_actions:
        :param create_net: should return {'pi': pi , 'v': v}; variables created for pi, v, status_encoder
        should be under scope '/pi', '/v', '/se'
        :param parent: parent network
        :type parent: ActorCritic
        :param kwargs:
        """
        self.num_actions, self.state_shape = num_actions, state_shape
        self.parent = parent  # global ActorCritic
        self.ddqn, self.aux_r, self.aux_d, self.reward_decay = ddqn, aux_r, aux_d, reward_decay
        self.optimizer = optimizer
        with tf.name_scope("input"):
            self.input_state = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name="input_state")
            self.input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
            self.input_value = tf.placeholder(dtype=tf.float32, shape=[None], name="input_value")
            self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None], name="input_reward")
            self.input_terminate = tf.placeholder(dtype=tf.float32, shape=[None], name="input_terminate")
            self.input_entropy = tf.placeholder(dtype=tf.float32, name="input_entropy")

        with tf.variable_scope(name):
            with tf.variable_scope("learn"):
                net = create_net(self.input_state, num_actions)
            self.pi_dist, self.q = net["pi"], net["q"]
            with tf.variable_scope("target"):
                target_net = create_net(self.input_state, num_actions)

            variable_pi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/learn/pi")
            variable_q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/learn/q")
            target_variable_q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/target/q")
            variable_se = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/learn/se")  # shared state encoder
            target_variable_se = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name+"/target/se")  # shared state encoder
            variable_r = []
            logging.warning("variable_pi:%s, variable_v: %s", variable_pi, variable_q)
            if self.aux_r:
                variable_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + "learn/r")
                self.r = net["r"]
            self.var_all = variable_se + variable_pi + variable_q + variable_r
            self.variable_se, self.variable_pi, self.variable_q, self.variable_r = variable_se, variable_pi, variable_q, variable_r
            with tf.variable_scope("global_grad"):
                self.acc_pi = [tf.get_variable(escape(v.name) + "grad", dtype=tf.float32,
                                                 initializer=tf.zeros_like(v.initialized_value())) for v in variable_pi]
                self.acc_q = [tf.get_variable(escape(v.name) + "grad", dtype=tf.float32,
                                                 initializer=tf.zeros_like(v.initialized_value())) for v in variable_q]
                self.acc_se = [tf.get_variable(escape(v.name) + "grad", dtype=tf.float32,
                                                 initializer=tf.zeros_like(v.initialized_value())) for v in variable_se]
                self.acc_r = [tf.get_variable(escape(v.name) + "grad", dtype=tf.float32,
                                             initializer=tf.zeros_like(v.initialized_value())) for v in variable_r]
                self.acc_on = self.acc_se + self.acc_pi + self.acc_q
                self.acc_off = self.acc_se + self.acc_q + self.acc_r
            var_on = variable_se + variable_pi + variable_q
            var_off = variable_se + variable_q + variable_r

            self.v = tf.reduce_max(self.q, axis=1)
            self.td = self.input_value - self.v

            with tf.name_scope("on_policy"):
                # train pi
                logpi = tf.log(self.pi_dist)
                self.entropy = -tf.reduce_sum(logpi * self.pi_dist, axis=1)
                logging.warning("input_action:%s, logpi:%s, td:%s, entropy:%s", self.input_action, logpi, self.td, self.entropy)
                action_onehot = tf.one_hot(self.input_action, num_actions, dtype=tf.float32)
                self.pi_loss = tf.reduce_mean(tf.reduce_sum(action_onehot * logpi, axis=1) * tf.stop_gradient(self.td)) \
                               + tf.reduce_mean(self.input_entropy * self.entropy)
                self.pi_loss = -self.pi_loss
                # train q
                self.v_loss = tf.reduce_mean(Network.clipped_square(self.td))
                self.on_loss = self.pi_loss + self.v_loss
                grad_on_local = tf.gradients(self.on_loss, var_on)
                with tf.control_dependencies(grad_on_local):
                    assigns_on = [tf.assign_add(g_grad, l_grad) for g_grad, l_grad in zip(self.acc_on, grad_on_local)]
                    self.summary_on = tf.summary.merge([
                        tf.summary.scalar("entropy", tf.reduce_mean(self.entropy)),
                        tf.summary.scalar("pi_loss", self.pi_loss),
                        tf.summary.scalar("td", tf.reduce_mean(self.td)),
                        tf.summary.scalar("grad_local_on", tf.add_n([tf.reduce_mean(g) for g in grad_on_local])),
                        tf.summary.scalar("grad_acc_on", tf.add_n([tf.reduce_mean(g) for g in assigns_on])),
                        tf.summary.scalar("v", tf.reduce_mean(self.v))
                    ])
                    self.compute_on_policy = assigns_on

            with tf.name_scope("off_policy"):
                # dqn: double dqn, compute target V
                if self.ddqn:
                    max_action = tf.argmax(self.q, axis=1)
                    max_action = tf.one_hot(max_action, num_actions, dtype=tf.float32)
                    target_q = tf.reduce_sum(target_net["q"] * max_action, axis=1)
                else:
                    target_q = tf.reduce_max(target_net["q"], axis=1)
                self.target_v = self.input_reward + (1.0 - self.input_terminate) * self.reward_decay * target_q
                self.summary_target_v = tf.summary.merge([
                    tf.summary.scalar("target_v", tf.reduce_mean(self.target_v)),
                    tf.summary.scalar("target_q", tf.reduce_mean(target_q)),
                ])

                # Q iteration, second sess.run() using target_v as input_value
                self.q_losses = tf.square(self.input_value - tf.reduce_sum(action_onehot * self.q, axis=1))
                self.off_losses = self.q_losses
                # aux R iteration
                self.r_losses = None
                if self.aux_r:
                    self.r_losses = tf.square(self.input_reward - self.r)
                    self.off_losses = self.off_losses + self.r_losses

                self.off_loss = tf.reduce_mean(self.off_losses)
                grad_off_local = tf.gradients(self.off_loss, var_off)
                with tf.control_dependencies(grad_off_local):
                    assigns_off = [tf.assign_add(g_grad, l_grad) for g_grad, l_grad in zip(self.acc_off, grad_off_local)]
                    self.summary_off = [
                        tf.summary.scalar("q", tf.reduce_mean(self.q)),
                        tf.summary.scalar("q_loss", tf.reduce_mean(self.q_losses)),
                        tf.summary.scalar("grad_local_off", tf.add_n([tf.reduce_mean(g) for g in grad_off_local])),
                        tf.summary.scalar("grad_acc_off", tf.add_n([tf.reduce_mean(g) for g in assigns_off])),
                    ]
                    if self.r_losses is not None:
                        self.summary_off.append(tf.summary.scalar("r_loss", tf.reduce_mean(self.r_losses)))
                    self.summary_off = tf.summary.merge(self.summary_off)
                    self.compute_off_policy = assigns_off

            with tf.name_scope("apply"):
                test_and_applies = []
                for g_grad, vars in [(self.acc_q, variable_q), (self.acc_pi, variable_pi),
                                     (self.acc_se, variable_se), (self.acc_r, variable_r)]:
                    if len(g_grad) == 0:
                        continue
                    apply_g = self.optimizer.apply_gradients(zip(g_grad, vars), global_step=global_step)
                    with tf.control_dependencies([apply_g]):
                        clears = [tf.assign(g, tf.zeros_like(g)) for g in g_grad]
                        with tf.control_dependencies(clears):
                            after_clear = tf.no_op()
                    test_and_apply = tf.cond(tf.equal(tf.add_n([tf.reduce_sum(g) for g in g_grad]), 0),
                                                  lambda: tf.no_op(), lambda: after_clear)
                    test_and_applies.append(test_and_apply)
                self.test_and_apply = test_and_applies

            with tf.name_scope("follow"):
                self.follows = [tf.assign(target, learn) for target, learn in
                                zip(target_variable_se + target_variable_q, variable_se + variable_q)]

            if self.parent is not None:
                with tf.name_scope("push"):
                    test_and_applies = []
                    for g_grad, vars in [(self.acc_q, self.parent.variable_q), (self.acc_pi, self.parent.variable_pi),
                                         (self.acc_se, self.parent.variable_se), (self.acc_r, self.parent.variable_r)]:
                        if len(g_grad) == 0:
                            continue
                        apply_g = self.parent.optimizer.apply_gradients(zip(g_grad, vars), global_step=global_step)
                        with tf.control_dependencies([apply_g]):
                            clears = [tf.assign(g, tf.zeros_like(g)) for g in g_grad]
                            with tf.control_dependencies(clears):
                                after_clear = tf.no_op()
                        test_and_apply = tf.cond(tf.equal(tf.add_n([tf.reduce_sum(g) for g in g_grad]), 0),
                                                      lambda: tf.no_op(), lambda: after_clear)
                        test_and_applies.append(test_and_apply)
                    self.push_and_apply = test_and_applies
                with tf.name_scope("pull"):
                    pulls = []
                    for v_local, v_parent in zip(self.var_all, self.parent.var_all):
                        pulls.append(tf.assign(v_local, v_parent))
                    self.pulls = pulls
            self.sess = None

    def set_session(self, sess):
        self.sess = sess

    def compute_on_gradient(self, state, action, reward, value, entropy):
        """

        :param state:
        :param action: vector [0,0,..,1,0,...] with action index = 1
        :param value:
        :return:
        """
        result = self.sess.run(self.compute_on_policy + [self.pi_loss, self.v_loss, self.td, self.entropy],
                               feed_dict={self.input_state: state,
                                          self.input_action: action,
                                          self.input_reward: reward,
                                          self.input_value: value,
                                          self.input_entropy: entropy})
        return result[-4:]

    def compute_off_gradient(self, state, action, reward, target_value, terminate):
        """

        :param state:
        :param action: vector [0,0,..,1,0,...] with action index = 1
        :param value:
        :return:
        """
        result = self.sess.run(self.compute_off_policy + [self.off_losses, self.summary_off],
                               feed_dict={self.input_state: state,
                                          self.input_action: action,
                                          self.input_reward: reward,
                                          self.input_value: target_value,
                                          self.input_terminate: terminate})
        return result[-2:]

    def apply_gradient(self):
        """
        apply gradients to local network
        :return:
        """
        self.sess.run(self.test_and_apply)

    def push_apply_gradient(self):
        """
        push gradients to parent, and apply on parent network
        :return:
        """
        self.sess.run(self.push_and_apply)

    def pull_weights(self):
        """
        pull weights from parent network
        :return:
        """
        self.sess.run(self.pulls)

    def get_action(self, state):
        return self.sess.run([self.pi_dist], feed_dict={self.input_state: state})[0]

    def get_v(self, state):
        return self.sess.run([self.v], feed_dict={self.input_state: state})[0]

    def get_q(self, state):
        return self.sess.run([self.q], feed_dict={self.input_state: state})[0]

    def get_target_v(self, state, reward, terminate):
        return self.sess.run([self.target_v, self.summary_target_v], feed_dict={self.input_state: state,
                                                         self.input_reward: reward,
                                                         self.input_terminate: terminate})

    def update_target(self):
        self.sess.run([self.follows])


class A3CAgent(hrl.tf_dependent.base.BaseDeepAgent):

    # def init_hyper_params(self, hyper_param):
    #     self.hyper_param = {"reward_decay": 0.99,
    #                         "replay_size": 10 ** 4,
    #                         "replay_state_scale": 1.0,
    #                         "replay_state_offset": 0,
    #                         "train_on_interval": 8,
    #                         "train_off_interval": 4,
    #                         "target_follow_interval": 1000,
    #                         # "noise_count": 20,
    #                         # "noise_scale": 1.0, "noise_decay": 0.99, "noise_decay_step": 1000000,
    #                         "learning_rate": 0.0001,
    #                         "off_batch_size": 32,
    #                         "L2": 0.01,
    #                         "max_gradient": 20.0,
    #                         "entropy": 0.01,
    #                         "min_stddev": 0.01,
    #                         "max_stddev": 4.0,
    #                         "max_mean": 0.98,
    #                         "playback": "NP",  # "NP", "Playback", "LSH", "PER" for Prioritized Experience Replay
    #                         "double_dqn": False,
    #                         "soft_follow": True,  # soft: target slowly follows learned network;
    #                         # hard: target reset to learned network periodically
    #                         "soft_follow_speed": 0.001,  # soft follow speed
    #                         "hard_follow_period": 500,  # train interval between target follow
    #                         "aux_r": False,  # auxiliary task predicting immediate reward R
    #                         "aux_d": False,  # auxiliary task predicting next status
    #                         "action_on": True,  # True to choose action according to on policy: pi; False to q
    #                         "e_greedy": 0.5,  # e-greedy
    #                         "ddqn": True,
    #                         }
    #     self.hyper_param.update(hyper_param)
    #     logging.warning("hyper param: %s", self.hyper_param)
    #     pass

    def __init__(self, create_net, state_shape, num_actions,
                 replay_capacity,
                 reward_decay,
                 train_on_interval,
                 train_off_interval,
                 target_follow_interval,
                 off_batch_size,
                 entropy,
                 playback_class=hrl.playback.MapPlayback,
                 replay_state_offset=0,
                 replay_state_scale=1,
                 optimizer=tf.train.AdamOptimizer(1e-4),
                 ddqn=False,
                 aux_r=False,
                 aux_d=False,
                 name=None, index=0, global_step=None, parent_net=None, **kwargs):
        kwargs.update({"entropy": entropy})
        super(A3CAgent, self).__init__(self, **kwargs)
        self.name, self.index = name, index
        self.state_shape, self.action_n = list(state_shape), num_actions
        self.reward_decay, self.train_on_interval, self.train_off_interval, self.target_follow_interval, \
        self.off_batch_size, self.entropy = reward_decay, train_on_interval, train_off_interval, \
                                            target_follow_interval, off_batch_size, entropy
        # playback buffer
        if train_on_interval > 0:
            self.replay_on = hrl.playback.MapPlayback(train_on_interval,
                                                      sample_shapes={
                                                          "state": self.state_shape,
                                                          "action": [],
                                                          "next_state": self.state_shape,
                                                          "reward": [],
                                                          "episode_done": []
                                                      }
                                                      , pop_policy="sequence"
                                                      , augment_offset={
                                                          "state": replay_state_offset,
                                                          "next_state": replay_state_offset,
                                                      }, augment_scale={
                                                          "state": replay_state_scale,
                                                          "next_state": replay_state_scale,
                                                      }
                                                      )
        else:
            self.replay_on = hrl.playback.MapPlayback(1,
                                                      sample_shapes={
                                                          "state": self.state_shape,
                                                          "action": [],
                                                          "next_state": self.state_shape,
                                                          "reward": [],
                                                          "episode_done": []
                                                      }, pop_policy="sequence"
                                                      , augment_offset={
                                                          "state": replay_state_offset,
                                                          "next_state": replay_state_offset,
                                                      }, augment_scale={
                                                          "state": replay_state_scale,
                                                          "next_state": replay_state_scale,
                                                      }
                                                      )

        self.replay_off = playback_class(replay_capacity,
                                         sample_shapes={
                                             "state": self.state_shape,
                                             "action": [],
                                             "next_state": self.state_shape,
                                             "reward": [],
                                             "episode_done": []
                                         }
                                         , augment_offset={
                                             "state": replay_state_offset,
                                             "next_state": replay_state_offset,
                                         }, augment_scale={
                                             "state": replay_state_scale,
                                             "next_state": replay_state_scale,
                                         }
                                         )

        self.step_n = 0
        # last state & action
        self.summary_writer = None
        self.dummy_input_signal = np.ndarray(shape=(1, 1), dtype=np.float32)
        self.sess = None
        self.global_step = global_step
        self.net = ActorCritic(index, "%s%d" % (name, index), state_shape, num_actions, create_net,
                               optimizer=optimizer, parent=parent_net, global_step=global_step,
                               ddqn=ddqn, aux_r=aux_r, aux_d=aux_d, reward_decay=reward_decay)
        pass

    def set_session(self, sess):
        super(A3CAgent, self).set_session(sess)
        self.net.set_session(sess)

    def post_initialize(self):
        # called after set_session
        self.net.update_target()

    def act(self, state, evaluate=False, **kwargs):
        self.step_n += 1
        action = self.net.get_action(np.asarray([state]))[0]  # batch size 1
        logging.warning("action: min:%s, max:%s, entropy: %s", np.min(action), np.max(action),
                        -np.sum(action * np.log(action)))

        last_action = np.random.choice(np.arange(self.action_n), p=action)
        # else:
        #     q = self.net.get_q(np.asarray([state]))[0]
        #     if explore:
        #         if np.random.rand() < self.hyper_param["e_greedy"]:
        #             # greedy
        #             q = np.exp(q)
        #             self.last_action = np.random.choice(np.arange(self.action_n), p=q / np.sum(q))
        #         else:
        #             # random
        #             self.last_action = np.argmax(q)
        #     else:
        #         self.last_action = np.argmax(q)

        return last_action

    def step(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        """
        save {s[t], a[t], r[t], s[t+1], terminate} to playback
        :param state: state
        :param action: action
        :param reward: reward observed
        :param next_state: next state
        :param episode_done: 0: not terminated; 1: terminated
        :return:
        """
        self._stepper.step()
        info = {}
        has_update = False
        if self.train_on_interval > 0:
            self.replay_on.push_sample(
                {
                    'state': state,
                    'action': np.asarray(action, dtype=np.uint8),
                    'next_state': next_state,
                    'reward': np.asarray(reward, dtype=np.float32),
                    'episode_done': np.asarray(episode_done, dtype=np.float32)
                }, reward
            )
            batch_size = self.replay_on.get_count()
            if batch_size >= self.train_on_interval or episode_done:
                has_update = True
                # sample random batch from playback
                batch = self.replay_on.sample_batch(batch_size)
                self.replay_on.reset()
                Si, Ai, Ri, Sj, T = np.asarray(batch['state'], dtype=float), batch['action'], batch['reward'], \
                                    np.asarray(batch['next_state'], dtype=float), batch['episode_done']

                R = np.zeros(shape=[batch_size], dtype=float)
                if episode_done:
                    r = 0.0
                else:
                    last_v = self.net.get_v(np.asarray([next_state]))
                    r = last_v[0]
                for i in range(batch_size):
                    index = batch_size - i - 1
                    if T[index] != 0:
                        # logging.warning("Terminated!, Ri:%s, Vi:%s", Ri[index], Vi[index])
                        r = 0
                    r = Ri[index] + self.reward_decay * r
                    R[index] = r
                target_name = "Terminate" if episode_done else "bootstrap"
                logging.warning("Target from %s: [ %s ... %s]", target_name, R[0], R[-1])

                # train V Pi, entropy annealing
                pi_loss, v_loss, td, entropy = self.net.compute_on_gradient(state=Si, action=Ai, reward=Ri, value=R,
                                                                            entropy=self.entropy)
                info.update({"on/target_v": R, "on/entropy": entropy, "on/entropy_param": self.entropy,
                             "on/pi_loss": pi_loss, "on/v_loss": v_loss, "on/td": td})

        if self.train_off_interval > 0:
            self.replay_off.push_sample(
                {
                    'state': state,
                    'action': np.asarray(action, dtype=np.uint8),
                    'next_state': next_state,
                    'reward': np.asarray(reward, dtype=np.float32),
                    'episode_done': np.asarray(episode_done, dtype=np.float32)
                }, reward
            )

            if self.step_n % self.train_off_interval == 0 \
                    and self.replay_off.get_count() > self.off_batch_size * 4:
                has_update = True
                # sample random batch from playback
                batch_index = self.replay_off.next_batch_index(self.off_batch_size)
                batch_size = len(batch_index)
                batch = self.replay_off.get_batch(batch_index)
                Si, Ai, Ri, Sj, T = np.asarray(batch['state'], dtype=float), batch['action'], batch['reward'], \
                                    np.asarray(batch['next_state'], dtype=float), batch['episode_done']

                # training for off policy data
                target_value, summary = self.net.get_target_v(state=Sj, reward=Ri, terminate=T)
                # todo for PER: should feed in sample weights
                loss, summary = self.net.compute_off_gradient(state=Si, action=Ai, reward=Ri,
                                                              target_value=target_value,
                                                              terminate=T)

                self.replay_off.update_score(batch_index, loss.reshape(batch_size, 1))

        if has_update:
            self.net.push_apply_gradient()
            self.net.pull_weights()
            # # apply gradient no matter what
            # if self.index == 0:
            #     self.net.apply_gradient()

        if self.target_follow_interval > 0 and self.step_n % self.target_follow_interval == 0:
            self.net.update_target()

        return None, info
