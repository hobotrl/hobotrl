# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl
import hobotrl.network as network
import hobotrl.sampling as sampling
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.playback import MapPlayback


class EpsilonGreedyPolicy(hrl.core.Policy):

    def __init__(self, q_function, epsilon, num_actions):
        super(EpsilonGreedyPolicy, self).__init__()
        self.q_function, self._epsilon, self._num_actions = q_function, epsilon, num_actions

    def act(self, state, **kwargs):
        if np.random.rand() < self._epsilon:
            # random
            return np.random.randint(self._num_actions)
        q_values = self.q_function(np.asarray([state]))
        action = np.argmax(q_values)
        return action


class ValueBasedAgent(hrl.core.Agent):

    def __init__(self, greedy_epsilon, num_actions, *args, **kwargs):
        kwargs.update({"greedy_epsilon": greedy_epsilon, "num_actions": num_actions})
        super(ValueBasedAgent, self).__init__(*args, **kwargs)
        self._q_function = self.init_value_function(*args, **kwargs)
        self._policy = self.init_policy(*args, **kwargs)

    def init_value_function(self, *args, **kwargs):
        """
        should be implemented by sub-classes.
        should return Q Function
        :param args:
        :param kwargs:
        :return:
        :rtype: network.Function
        """
        raise NotImplementedError()

    def init_policy(self, greedy_epsilon, num_actions, *args, **kwargs):
        return EpsilonGreedyPolicy(self._q_function, greedy_epsilon, num_actions)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)


class L2(network.NetworkUpdater):

    def __init__(self, net_or_var_scope):
        super(L2, self).__init__()
        if isinstance(net_or_var_scope, network.Network):
            var_scope = net_or_var_scope.relative_var_scope
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
        return network.UpdateRun()


class OneStepTD(network.NetworkUpdater):

    def __init__(self, learn_q, target_q, num_actions, discount_factor=0.99, ddqn=False):
        super(OneStepTD, self).__init__()
        self._f_learn_q, self._f_target_q = learn_q, target_q
        self._input_target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_q")
        self._input_action = tf.placeholder(dtype=tf.uint8, shape=[None], name="input_action")
        self._discount_factor, self._ddqn, self._num_actions = discount_factor, ddqn, num_actions
        op_q = learn_q.output().op
        one_hot = tf.one_hot(self._input_action, num_actions)
        selected_q = tf.reduce_sum(one_hot * op_q, axis=1)
        self._sym_loss = tf.reduce_mean(
            network.Utils.clipped_square(
                self._input_target_q - selected_q
            )
        )
        print "selected_q:", selected_q, ", sym_loss:", self._sym_loss, "one_hot:", one_hot,", op:", op_q
        self._update_operation = network.MinimizeLoss(self._sym_loss, var_list=self._f_learn_q.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        if not self._ddqn:
            target_q_val = self._f_target_q(batch["next_state"])
            target_q_val = np.max(target_q_val, axis=1)
        else:
            learn_q_val = self._f_learn_q(batch["next_state"])
            target_action = np.argmax(learn_q_val, axis=1)
            target_q_val = np.sum(self._f_target_q(batch["next_state"]) * hrl.utils.NP.one_hot(target_action, self._num_actions), axis=1)
        target_q_val = batch["reward"] + self._discount_factor * target_q_val * (1.0 - batch["episode_done"])
        feed_dict = {self._input_target_q: target_q_val, self._input_action: batch["action"]}
        feed_dict.update(self._f_learn_q.input_dict(batch["state"]))
        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={"target_q": target_q_val, "td_loss": self._sym_loss})


class DQN(sampling.TransitionBatchUpdate,
          ValueBasedAgent,
          BaseDeepAgent):
    def __init__(self,
                 f_create_q, state_shape,
                 # OneStepTD arguments
                 num_actions, discount_factor, ddqn,
                 # target network sync arguments
                 target_sync_interval,
                 target_sync_rate,
                 # epsilon greeedy arguments
                 greedy_epsilon,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # sampler arguments
                 update_interval=4, replay_size=1000, batch_size=32,
                 sampler=None,
                 *args, **kwargs):
        kwargs.update({
            "f_create_q": f_create_q,
            "state_shape": state_shape,
            "num_actions": num_actions,
            "discount_factor": discount_factor,
            "ddqn": ddqn,
            "target_sync_interval": target_sync_interval,
            "target_sync_rate": target_sync_rate,
            "update_interval": update_interval,
            "replay_size": replay_size,
            "batch_size": batch_size,
            "greedy_epsilon": greedy_epsilon,
            "max_gradient": max_gradient
        })
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        if sampler is None:
            sampler = sampling.TransitionSampler(MapPlayback(replay_size), batch_size, update_interval)
            kwargs.update({"sampler": sampler})
        # call super.__init__
        super(DQN, self).__init__(*args, **kwargs)

        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(OneStepTD(self.learn_q, self.target_q, num_actions, discount_factor, ddqn), name="td")
        network_optimizer.add_updater(L2(self.network), name="l2")
        network_optimizer.compile()
        self._target_sync_interval, self._target_sync_rate = target_sync_interval, target_sync_rate

    def init_network(self, f_create_q, state_shape, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        return network.NetworkWithTarget([input_state], f_create_q, var_scope="learn", target_var_scope="target")

    def init_value_function(self, **kwargs):
        self.learn_q = network.NetworkFunction(self.network["q"])
        self.target_q = network.NetworkFunction(self.network.target["q"])
        return self.learn_q

    def update_on_transition(self, batch):
        self.network_optimizer.updater("td").update(self.sess, batch)
        self.network_optimizer.updater("l2").update(self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self.step_n % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {}

    def set_session(self, sess):
        super(DQN, self).set_session(sess)
        self.learn_q.set_session(sess)
        self.target_q.set_session(sess)


if __name__ == '__main__':
    import gym
    import hobotrl.environments as envs

    env = gym.make('Pendulum-v0')
    env = hrl.envs.C2DEnvWrapper(env, [5])
    env = hrl.envs.ScaledRewards(env, 0.1)
    state_shape = list(env.observation_space.shape)
    global_step = tf.get_variable(
        'global_step', [], dtype=tf.int32,
        initializer=tf.constant_initializer(0), trainable=False
    )

    def f_q(inputs):
        q = network.Utils.layer_fcs(inputs[0], [200, 100], env.action_space.n, l2=1e-4)
        return {"q": q}

    agent = DQN(
        f_create_q=f_q,
        state_shape=state_shape,
        # OneStepTD arguments
        num_actions=env.action_space.n,
        discount_factor=0.99,
        ddqn=False,
        # target network sync arguments
        target_sync_interval=100,
        target_sync_rate=1.0,
        # sampler arguments
        update_interval=4, replay_size=1000, batch_size=32,
        # epsilon greedy arguments
        greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.5, 0.1),
        global_step=global_step,
        network_optmizer=network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0)
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = agent.init_supervisor(
        graph=tf.get_default_graph(), worker_index=0,
        init_op=tf.global_variables_initializer(), save_dir="dqn_new"
    )
    with sv.managed_session(config=config) as sess:
        agent.set_session(sess)
        runner = hrl.envs.EnvRunner(
            env, agent, evaluate_interval=sys.maxint,
            render_interval=sys.maxint, logdir="dqn_new"
        )
        runner.episode(500)