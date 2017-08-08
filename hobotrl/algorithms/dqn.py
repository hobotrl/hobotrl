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
from hobotrl.policy import EpsilonGreedyPolicy
import hobotrl.target_estimate as target_estimate


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
        if ddqn:
            estimator = target_estimate.DDQNOneStepTD(self.learn_q, self.target_q, discount_factor)
        else:
            estimator = target_estimate.OneStepTD(self.target_q, discount_factor)
        network_optimizer.add_updater(network.FitTargetQ(self.learn_q, estimator), name="td")
        network_optimizer.add_updater(network.L2(self.network), name="l2")
        network_optimizer.compile()
        self._target_sync_interval, self._target_sync_rate = target_sync_interval, target_sync_rate
        self._update_count = 0

    def init_network(self, f_create_q, state_shape, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        return network.NetworkWithTarget([input_state], f_create_q, var_scope="learn", target_var_scope="target")

    def init_value_function(self, **kwargs):
        self.learn_q = network.NetworkFunction(self.network["q"])
        self.target_q = network.NetworkFunction(self.network.target["q"])
        return self.learn_q

    def update_on_transition(self, batch):
        self._update_count += 1
        self.network_optimizer.updater("td").update(self.sess, batch)
        self.network_optimizer.updater("l2").update(self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {"score": info["td_losses"]}

    def set_session(self, sess):
        super(DQN, self).set_session(sess)
        self.learn_q.set_session(sess)
        self.target_q.set_session(sess)

