# -*- coding: utf-8 -*-
# Reinforcement Learning with Deep Energy-Based Policies https://arxiv.org/abs/1702.08165
# https://github.com/haarnoja/softqlearning

import sys
import logging
import tensorflow as tf
import numpy as np
from hobotrl import network, DQN

from hobotrl.core import Policy
from hobotrl.network import Network, NetworkUpdater, NetworkFunction, MinimizeLoss, UpdateRun, LocalOptimizer, \
    NetworkWithTarget, FitTargetQ, L2, Function
from hobotrl.policy import WrapEpsilonGreedy
from hobotrl.target_estimate import TargetEstimator, ContinuousActionEstimator
from hobotrl.tf_dependent.base import BaseDeepAgent
from playground.mpc import ModelUpdater, MPCPolicy


class SVGDUpdater(NetworkUpdater):
    def __init__(self, generator_net, logprob_net, m_particles=16, alpha_exploration=1.0):
        """
        SVGD, updates generator_net to match PDF of logprob_net.
         Using unit gaussian kernel.
        :param generator_net: generator_net(state, noise) => action
        :type generator_net: Network
        :param logprob_net: logprob_net(state, action) => log pdf of action
        :type logprob_net: Network
        """
        super(SVGDUpdater, self).__init__()
        self._m_particles = m_particles
        self._alpha_exploration = alpha_exploration
        m = m_particles
        h_epsilon = 1e-3
        self._generator_net, self._logprob_net = generator_net, logprob_net
        state_shape = generator_net.inputs[0].shape.as_list()
        noise_shape = generator_net.inputs[1].shape.as_list()
        action_shape = logprob_net.inputs[1].shape.as_list()
        dim_state = state_shape[1]
        self._dim_noise = noise_shape[1]
        dim_action = action_shape[1]
        with tf.name_scope("inputs"):
            self._input_state = tf.placeholder(tf.float32, state_shape, "input_state")
            self._input_noise = tf.placeholder(tf.float32, noise_shape, "input_noise")
            self._input_alpha = tf.placeholder(tf.float32, [], "input_alpha")
        with tf.name_scope("generate"):
            state_batch = tf.shape(self._input_state)[0]
            state = tf.reshape(tf.tile(
                tf.reshape(self._input_state, shape=[-1, 1, dim_state])
                , (1, m, 1))
                , shape=[-1, dim_state])
            noise = tf.tile(self._input_noise, (state_batch, 1))
            # generate action with tuple:
            #   (s0, n0), (s0, n1), ..., (s1, n0), (s1, n1), ...
            generator_net = generator_net([state, noise], name_scope="batch_generator")
            for name in generator_net.outputs:
                actions = generator_net[name].op
                break
            # actions: [bs * m, dim_a]
            action_square = tf.tile(tf.reshape(actions, [-1, 1, m, dim_action]), (1, m, 1, 1))
            # sub: [b_s, m, m, dim_a]
            action_sub = tf.transpose(action_square, perm=[0, 2, 1, 3]) - action_square
            # dis square: [b_s, m, m]
            dis_square = tf.reduce_sum(tf.square(action_sub), axis=3)

            # h: [b_s]
            # h from median
            # median_square, _ = tf.nn.top_k(tf.reshape(dis_square, [-1, m * m]), m * m // 2 + 1, True)
            # median_square = median_square[:, -1]
            # h = tf.sqrt(median_square)

            # h from mean
            h = tf.reduce_mean(tf.sqrt(dis_square), axis=(1, 2))

            # h = h / (2 * np.log(m + 1))
            # h = h**2 / (2 * np.log(m + 1))
            h = h**2 / (np.log(m + 1))  # more stable
            h = h + h_epsilon
            self._h = h
            # k: [bs, m, m]
            k = tf.exp(-1.0 / tf.reshape(h, (-1, 1, 1)) * dis_square)
            # dk: [bs, m, m, dim_a]
            dk = tf.reshape(k, (-1, m, m, 1)) * (2 / tf.reshape(h, (-1, 1, 1, 1))) * action_sub
            # dlogprob: [bs, m, 1]
            logprob_net = logprob_net([state, actions], name_scope="batch_logprob")
            for name in logprob_net.outputs:
                action_logprob = logprob_net[name].op
                break
            dlogp = tf.gradients(action_logprob, actions)
            # dlogp/da: [bs, m, m, dim_a]
            dlogp_matrix = tf.tile(tf.reshape(dlogp, (state_batch, 1, m, dim_action)), (1, m, 1, 1))
            # svgd gradient: [bs, m, m, dim_a]
            grad_svgd = tf.reshape(k, (-1, m, m, 1)) * dlogp_matrix + dk * self._input_alpha
            # [bs, m, dim_a]
            grad_svgd = tf.reduce_mean(grad_svgd, axis=2)
            # [bs * m, dim_a]
            grad_svgd = tf.reshape(grad_svgd, (-1, dim_action))
            generator_loss = tf.reduce_mean(-tf.stop_gradient(grad_svgd) * actions)
            self._loss = generator_loss
            self._grad_svgd = grad_svgd
            self._op = MinimizeLoss(self._loss, var_list=generator_net.variables)

    def declare_update(self):
        return self._op

    def update(self, sess, batch, *args, **kwargs):
        state = batch["state"]
        noise = np.random.normal(0, 1, (self._m_particles, self._dim_noise))
        return UpdateRun(feed_dict={self._input_state: state,
                                    self._input_noise: noise,
                                    self._input_alpha: self._alpha_exploration},
                         fetch_dict={
                             "loss": self._loss,
                             "svgd_grad": self._grad_svgd,
                             "h": self._h
                         })


class CriticUpdater(NetworkUpdater):

    def __init__(self, critic_net, target_estimator):
        self._critic_net = critic_net
        self._target_estimator = target_estimator
        super(CriticUpdater, self).__init__()
        self._input_target_q = tf.placeholder(tf.float32, [None], name="target_q")
        self._loss = network.Utils.clipped_square(critic_net["q"].op - self._input_target_q)
        self._op = MinimizeLoss(self._loss, var_list=critic_net.variables)

    def declare_update(self):
        return self._op

    def update(self, sess, batch, *args, **kwargs):
        target_value = self._target_estimator.estimate(**batch)
        return UpdateRun(feed_dict={
            self._critic_net.inputs[0]: batch["state"],
            self._critic_net.inputs[1]: batch["action"],
            self._input_target_q: target_value
        }, fetch_dict={
            "loss": self._loss,
            "action": batch["action"],
            "target_value": target_value
        })


class SoftVFunction(Function):
    def __init__(self, q_func, actor_func=None, m_particles=16, alpha_exploration=1.0):
        self._q_func, self._actor_func = q_func, actor_func
        self._m_particles = m_particles
        self._alpha_exploration = alpha_exploration
        if actor_func is not None:
            noise_shape = actor_func.inputs[1].shape.as_list()
            self._dim_noise = noise_shape[1]
        else:
            self._dim_noise = None
        self._dim_action = q_func.inputs[1].shape.as_list()[1]
        super(SoftVFunction, self).__init__()

    def __call__(self, *args, **kwargs):
        next_state = args[0]
        b_s, m, = next_state.shape[0], self._m_particles
        next_states = np.repeat(next_state, m, axis=0)
        if self._actor_func is not None:
            # sample action according to {actor_func} distribution
            noises = np.random.normal(0, 1, (b_s * m, self._dim_noise))
            next_actions = self._actor_func(next_states, noises)
        else:
            # sample action uniformly
            next_actions = np.random.uniform(-1, 1, [b_s * m, dim_action])
        # Equation 10
        next_qs = self._q_func(next_states, next_actions).reshape((b_s, m))
        next_value = self._alpha_exploration * \
                     np.log(np.average(np.exp(1.0 / self._alpha_exploration * next_qs), axis=1))
        return next_value

#
# class SoftVEstimator(TargetEstimator):
#
#     def __init__(self, v_func, discount_factor=0.99):
#
#         self._q_func, self._actor_func = q_func, actor_func
#         self._m_particles = m_particles
#         self._alpha_exploration = alpha_exploration
#         if actor_func is not None:
#             noise_shape = actor_func.inputs[1].shape.as_list()
#             self._dim_noise = noise_shape[1]
#         else:
#             self._dim_noise = None
#         self._dim_action = q_func.inputs[1].shape.as_list()[1]
#         super(SoftVEstimator, self).__init__(discount_factor)
#
#     def estimate(self, state, action, reward, next_state, episode_done, **kwargs):
#         target_value = reward + self._discount_factor * next_value * (1.0 - episode_done)
#         # Importance weights add just a constant to the value.
#         # next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
#         # next_value += self._action_dim * np.log(2)
#         return target_value


class SoftStochasticPolicy(Policy):
    def __init__(self, actor_func):
        self._actor_func = actor_func
        self._dim_noise = actor_func.inputs[1].shape.as_list()[1]
        super(SoftStochasticPolicy, self).__init__()

    def act(self, state, **kwargs):
        noise = np.random.normal(0, 1, [1, self._dim_noise])
        action = self._actor_func(np.array([state]), noise)[0]
        logging.warning("noise:%s, action:%s", noise, action)
        return action


class SoftQLearning(DQN):

    def __init__(self, f_create_actor, f_create_q,
                 state_shape, num_actions, dim_noise,
                 discount_factor, target_sync_interval,
                 target_sync_rate,
                 alpha_exploration=1.0,
                 network_optimizer=None,
                 max_gradient=10.0,
                 update_interval=4,
                 m_particle_svgd=16,
                 m_particle_v=16,
                 replay_size=1000, batch_size=32, sampler=None, *args, **kwargs):
        kwargs.update({
            "f_create_actor": f_create_actor,
            "dim_noise": dim_noise,
            "m_particle_svgd": m_particle_svgd,
            "m_particle_v": m_particle_v,
            "alpha_exploration": alpha_exploration,
        })
        if "ddqn" not in kwargs:
            kwargs["ddqn"] = False
        if "greedy_epsilon" not in kwargs:
            kwargs["greedy_epsilon"] = 1.0
        self._m_particle_svgd = m_particle_svgd
        self._m_particle_v = m_particle_v
        self._dim_noise = dim_noise
        self._alpha_exploration = alpha_exploration
        super(SoftQLearning, self).__init__(f_create_q, state_shape, num_actions, discount_factor,
                                            target_sync_interval=target_sync_interval,
                                            target_sync_rate=target_sync_rate,
                                            network_optimizer=network_optimizer,
                                            max_gradient=max_gradient, update_interval=update_interval,
                                            replay_size=replay_size,
                                            batch_size=batch_size, sampler=sampler,
                                            *args,
                                            **kwargs)

    def init_network(self, f_create_q, f_create_actor, state_shape, num_actions, dim_noise, *args, **kwargs):
        def f(inputs):
            state, action, noise = inputs
            actor_net = Network([state, noise], f_create_actor, var_scope="actor")
            q_net = NetworkWithTarget([state, action], f_create_q, var_scope="q", target_var_scope="target_q")
            return {
                "action": actor_net["action"],
                "q": q_net["q"],
                "target_q": q_net.target["q"]
            }, {
                "actor": actor_net,
                "q": q_net
            }
        with tf.name_scope("inputs"):
            input_state = tf.placeholder(tf.float32, [None, state_shape[0]], "state")
            input_action = tf.placeholder(tf.float32, [None, num_actions], "action")
            input_noise = tf.placeholder(tf.float32, [None, dim_noise], "noise")
        return Network([input_state, input_action, input_noise], f, var_scope="soft_q_learning")

    def init_value_function(self, **kwargs):
        actor_net = self.network.sub_net("actor")
        self._actor_func = NetworkFunction(actor_net["action"])
        self._target_q = NetworkFunction(self.network.sub_net("q").target["q"])
        self._learn_q = NetworkFunction(self.network.sub_net("q")["q"])
        self._target_v = SoftVFunction(self._target_q, self._actor_func, self._m_particle_v, self._alpha_exploration)
        return self._learn_q

    def init_policy(self, greedy_epsilon, num_actions, *args, **kwargs):
        return SoftStochasticPolicy(self._actor_func)

    def init_updaters_(self):
        estimator = ContinuousActionEstimator(self._target_v, discount_factor=self._discount_factor)
        self.network_optimizer.add_updater(CriticUpdater(self.network.sub_net("q"), estimator), name="td")
        self.network_optimizer.add_updater(SVGDUpdater(self.network.sub_net("actor"),
                                                       self.network.sub_net("q"),
                                                       m_particles=self._m_particle_svgd,
                                                       alpha_exploration=self._alpha_exploration), name="svgd")
        self.network_optimizer.add_updater(L2(self.network), name="l2")
        self.network_optimizer.compile()

    def update_on_transition(self, batch):
        self._update_count += 1
        self.network_optimizer.update("td", self.sess, batch)
        self.network_optimizer.update("svgd", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sub_net("q").sync_target(self.sess, self._target_sync_rate)
        return info, {"score": info["CriticUpdater/td/loss"]}


class SoftQMPC(SoftQLearning):

    def __init__(self, f_create_actor, f_create_q, f_model, state_shape, num_actions, dim_noise, discount_factor,
                 target_sync_interval, target_sync_rate,
                 greedy_epsilon=0.2, sample_n=4, horizon_n=4,
                 alpha_exploration=1.0,
                 network_optimizer=None, max_gradient=10.0, update_interval=4,
                 m_particle_svgd=16, m_particle_v=16, replay_size=1000, batch_size=32, sampler=None, *args, **kwargs):
        kwargs.update({
            "f_model": f_model,
            "greedy_epsilon": greedy_epsilon,
            "sample_n": sample_n,
            "horizon_n": horizon_n
        })
        self._sample_n, self._horizon_n = sample_n, horizon_n
        super(SoftQMPC, self).__init__(f_create_actor, f_create_q, state_shape, num_actions, dim_noise, discount_factor,
                                       target_sync_interval, target_sync_rate, alpha_exploration,
                                       network_optimizer, max_gradient,
                                       update_interval, m_particle_svgd, m_particle_v, replay_size, batch_size, sampler,
                                       *args, **kwargs)

    def init_network(self, f_create_q, f_create_actor, f_model, state_shape, num_actions, dim_noise, *args, **kwargs):
        def f(inputs):
            state, action, noise = inputs
            actor_net = Network([state, noise], f_create_actor, var_scope="actor")
            q_net = NetworkWithTarget([state, action], f_create_q, var_scope="q", target_var_scope="target_q")
            model_net = Network([state, action], f_model, var_scope="model")
            return {
                "action": actor_net["action"],
                "q": q_net["q"],
                "target_q": q_net.target["q"]
            }, {
                "actor": actor_net,
                "q": q_net,
                "model": model_net
            }
        with tf.name_scope("inputs"):
            input_state = tf.placeholder(tf.float32, [None, state_shape[0]], "state")
            input_action = tf.placeholder(tf.float32, [None, num_actions], "action")
            input_noise = tf.placeholder(tf.float32, [None, dim_noise], "noise")
        return Network([input_state, input_action, input_noise], f, var_scope="soft_q_learning")

    def init_policy(self, greedy_epsilon, num_actions, *args, **kwargs):

        # return WrapEpsilonGreedy(
        #     MPCPolicy(
        #         NetworkFunction({"goal": self.network.sub_net("model")["goal"],
        #                          "reward": self.network.sub_net("model")["reward"]}),
        #         actor_func=self._actor_func,
        #         value_func=self._target_v,
        #         sample_n=self._sample_n, horizon_n=self._horizon_n,
        #     )
        # , epsilon=greedy_epsilon, num_actions=num_actions, is_continuous=True)
        return MPCPolicy(
            NetworkFunction({"goal": self.network.sub_net("model")["goal"],
                                 "reward": self.network.sub_net("model")["reward"]}),
                actor_func=self._actor_func,
                value_func=self._target_v,
                sample_n=self._sample_n, horizon_n=self._horizon_n,
                alpha_exploration=self._alpha_exploration
            )

    def init_updaters_(self):
        estimator = ContinuousActionEstimator(self._target_v, self._discount_factor)
        self.network_optimizer.add_updater(CriticUpdater(self.network.sub_net("q"), estimator), name="td")
        self.network_optimizer.add_updater(SVGDUpdater(self.network.sub_net("actor"),
                                                       self.network.sub_net("q"),
                                                       m_particles=self._m_particle_svgd,
                                                       alpha_exploration=self._alpha_exploration), name="svgd")
        self.network_optimizer.add_updater(ModelUpdater(self.network.sub_net("model")), name="model")
        self.network_optimizer.add_updater(L2(self.network), name="l2")
        self.network_optimizer.compile()

    def update_on_transition(self, batch):
        self._update_count += 1
        self.network_optimizer.update("td", self.sess, batch)
        self.network_optimizer.update("svgd", self.sess, batch)
        self.network_optimizer.update("model", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sub_net("q").sync_target(self.sess, self._target_sync_rate)
        return info, {"score": info["CriticUpdater/td/loss"]}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dim_action = 2
    dim_noise = 2
    dim_state = 4

    def f_generator(inputs):
        state, noise = inputs[0], inputs[1]
        se = tf.concat([state, noise], axis=1)
        action = network.Utils.layer_fcs(se, [256, 256, 256], dim_action, activation_out=None, var_scope="action")
        return {"action": action}

    def f_normal(inputs):
        state, action = inputs[0], inputs[1]
        se = tf.concat([state, action], axis=1)
        mean = np.array([[0.5, 0.5]])
        return {"logp": - tf.reduce_sum(tf.square(action - mean), axis=1)}

    def f_normal2(inputs):
        action = inputs[1]
        mean1 = np.array([[1.0, 1.0]])
        mean2 = np.array([[-1.0, -1.0]])
        n1 = tf.exp(-tf.reduce_sum(tf.square(action - mean1), axis=1))
        n2 = tf.exp(-tf.reduce_sum(tf.square(action - mean2), axis=1))
        logp = tf.log(n1 + n2)
        return {"logp": logp}
    
    f_q = f_normal2
    input_state = tf.placeholder(dtype=tf.float32, shape=[None, dim_state], name="input_state")
    input_action = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name="input_action")
    input_noise = tf.placeholder(dtype=tf.float32, shape=[None, dim_noise], name="input_noise")
    generator_net = Network([input_state, input_noise], f_generator, var_scope="generator")
    generator_func = NetworkFunction(generator_net["action"])
    logprob_net = Network([input_state, input_action], f_q, var_scope="q")
    optimizer = LocalOptimizer()
    batch_size = 8
    m_particles = 128
    test_batch = 256
    optimizer.add_updater(SVGDUpdater(generator_net, logprob_net, m_particles=m_particles), name="svgd")
    optimizer.compile()
    plt.ion()
    plt.show()
    for g in range(8):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            generator_func.set_session(sess)
            for i in range(1000):
                state = np.random.normal(0, 1, (batch_size, dim_state))
                optimizer.update("svgd", sess, {"state": state})
                info = optimizer.optimize_step(sess)
                if i % 20 == 0:
                    state = np.random.normal(0, 1, (1, dim_state))
                    state = np.repeat(state, test_batch, axis=0)
                    noise = np.random.normal(0, 1, (test_batch, dim_noise))
                    actions = generator_func(state, noise)
                    coord = np.transpose(actions)
                    plt.clf()
                    plt.axis([-4, 4, -4, 4])
                    plt.plot(coord[0], coord[1], 'ro')
                    plt.draw()
                    plt.pause(0.01)

