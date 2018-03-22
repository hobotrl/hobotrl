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
from value_based import ValueBasedAgent, GreedyStateValueFunction
import hobotrl.target_estimate as target_estimate


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
        """
        :param f_create_q: function, f_create_q([state, action]) => {"q": op_q}
        :param state_shape: shape of state
        :param num_actions: action count
        :param discount_factor:
        :param ddqn: True if using double DQN
        :param target_sync_interval: interval syncing weights from learned network to target network
        :param target_sync_rate: syncing rate. 1.0 for hard sync, 0 < r < 1.0 for soft sync.
        :param greedy_epsilon: epsilon for epsilon greedy policy
        :param network_optimizer: NetworkOptimizer instance, default to LocalOptimizer
        :type network_optimizer: network.NetworkOptimizer
        :param max_gradient: gradient clip value
        :param update_interval: network update interval between Agent.step()
        :param replay_size: replay memory size.
        :param batch_size:
        :param sampler: Sampler, default to TransitionSampler.
                if None, a TransitionSampler is created using update_interval, replay_size, batch_size
        :param args:
        :param kwargs:
        """
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
        self._ddqn, self._discount_factor = ddqn, discount_factor
        self.init_updaters_()
        self._target_sync_interval, self._target_sync_rate = target_sync_interval, target_sync_rate
        self._update_count = 0

    def init_network(self, f_create_q, state_shape, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        return network.NetworkWithTarget([input_state], f_create_q, var_scope="learn", target_var_scope="target")

    def init_value_function(self, **kwargs):
        self.learn_q = network.NetworkFunction(self.network["q"])
        self.target_q = network.NetworkFunction(self.network.target["q"])
        self.target_v = GreedyStateValueFunction(self.target_q)
        return self.learn_q

    def init_updaters_(self):
        if self._ddqn:
            estimator = target_estimate.DDQNOneStepTD(self.learn_q, self.target_q, self._discount_factor)
        else:
            estimator = target_estimate.OneStepTD(self.target_q, self._discount_factor)
        self.network_optimizer.add_updater(network.FitTargetQ(self.learn_q, estimator), name="td")
        self.network_optimizer.add_updater(network.L2(self.network), name="l2")
        self.network_optimizer.compile()
        pass

    def update_on_transition(self, batch):
        self._update_count += 1
        self.network_optimizer.update("td", self.sess, batch)
        self.network_optimizer.update("l2", self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {"score": np.abs(info["FitTargetQ/td/td"])}

