# -*- coding: utf-8 -*-
# model predictive control
#

import logging
import numpy as np
import tensorflow as tf
from hobotrl import network, sampling

from hobotrl.core import Policy
from hobotrl.network import NetworkFunction, Network, Function, NetworkUpdater, MinimizeLoss, UpdateRun
from hobotrl.playback import MapPlayback
from hobotrl.policy import WrapEpsilonGreedy
from hobotrl.tf_dependent.base import BaseDeepAgent


class RandomDiscrete(Function):
    def __init__(self, n):
        self._n = n
        super(RandomDiscrete, self).__init__()

    def __call__(self, *args, **kwargs):
        state = args[0]
        batch = state.shape[0]
        return np.random.randint(0, self._n, batch)


class RandomContinuous(Function):
    def __init__(self, n):
        self._n = n
        super(RandomContinuous, self).__init__()

    def __call__(self, *args, **kwargs):
        state = args[0]
        batch = state.shape[0]
        return (np.random.rand(batch, self._n) - 0.5) * 2


class MPCPolicy(Policy):

    def __init__(self, model_func, actor_func=None, value_func=None, sample_n=4, horizon_n=4, dim_action=None):
        """
        :param model_func: transition model function. signature:
            model_func(state, action) => {'goal': goal, 'reward': reward}
        :type model_func: NetworkFunction
        :param actor_func: actor function, for action sampling. supports two type of actor function signature:
            1) actor_func(state) => action_distribution;
            2) actor_func(state, noise) => action
            if None, uniform sampling is used
        :type actor_func: NetworkFunction
        :param value_func: value function, for estimating future state value at rollout horizon.
            signature: value_func(state) => V
        :type value_func: NetworkFunction
        :param sample_n: number of actions to sample
        :param horizon_n: rollout steps
        :param dim_action: total candidate count for discrete action
        """
        super(MPCPolicy, self).__init__()
        self._actor_func, self._model_func, self._sample_n, self._horizon_n = actor_func, model_func, sample_n, horizon_n
        self._value_func = value_func
        action_shape = model_func.inputs[1].shape.as_list()
        self._is_continuous = len(action_shape) == 2
        if self._is_continuous:
            self._dim_action = action_shape[1]
        elif dim_action is not None:
            self._dim_action = dim_action
        if self._actor_func is not None and len(actor_func.inputs) == 2:
            self._dim_noise = actor_func.inputs[1].shape.as_list()[1]
        else:
            self._dim_noise = None
        # create default actor
        if self._is_continuous:
            self._default_actor = RandomContinuous(self._dim_action)
        else:
            self._default_actor = RandomDiscrete(self._dim_action)

    def act(self, state, **kwargs):
        init_s = np.asarray([state])
        if "random" in kwargs and kwargs["random"] is True:
            # return random action directly
            return self._default_actor(init_s)[0]
        actor = self._actor_func if self._actor_func is not None else self._default_actor
        actions, rewards = [], []
        for i in range(self._sample_n):
            reward = 0
            s = init_s
            for j in range(self._horizon_n):
                a = actor(s) if self._dim_noise is None else actor(s, np.random.normal(0, 1.0, (1, self._dim_noise)))
                if j == 0:
                    actions.append(a)
                logging.warning("i:%s,j:%s,s:%s,a:%s", i, j, s, a)
                next_s_r = self._model_func(s, a)
                g, r = next_s_r["goal"], next_s_r["reward"][0]
                s = s + g
                reward += r
                if j == self._horizon_n - 1 and self._value_func is not None:
                    reward += self._value_func(s)[0]
            rewards.append(reward)
        index = np.argmax(rewards)
        action = actions[index][0]
        logging.warning("reward:%s, index:%s, action%s", rewards, index, action)
        return action


class ModelUpdater(NetworkUpdater):
    def __init__(self, net_model, reward_weight=1.0):
        super(ModelUpdater, self).__init__()
        self._model = net_model(name_scope="off_model")
        goal, reward = self._model["goal"].op, self._model["reward"].op
        self._input_goal = tf.placeholder(tf.float32, goal.shape.as_list(), name="input_goal")
        self._input_reward = tf.placeholder(tf.float32, reward.shape.as_list(), name="input_reward")
        self._reward_loss = tf.reduce_mean(network.Utils.clipped_square(self._input_reward - reward))
        self._goal_loss = tf.reduce_mean(tf.reduce_sum(network.Utils.clipped_square(self._input_goal - goal), axis=1))
        self._loss = tf.reduce_mean(self._reward_loss * reward_weight + self._goal_loss)
        self._op = MinimizeLoss(self._loss, net_model.variables)

    def declare_update(self):
        return self._op

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state = batch["state"], batch["action"], batch["reward"], batch["next_state"]
        goal = next_state - state
        return UpdateRun(feed_dict={
            self._model.inputs[0]: state,
            self._model.inputs[1]: action,
            self._input_reward: reward,
            self._input_goal: goal
        }, fetch_dict={
            "loss_goal": self._goal_loss,
            "loss_reward": self._reward_loss,
            "action": action,
            "reward": reward,
            "state": state
        })


class MPCAgent(sampling.TransitionBatchUpdate,
               BaseDeepAgent):
    def __init__(self, f_model, sample_n, horizon_n, dim_state, dim_action, greedy_epsilon,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # sampler arguments
                 update_interval=4, replay_size=1000, batch_size=32,
                 sampler=None,
                 **kwargs):
        kwargs.update({
            "sample_n": sample_n,
            "horizon_n": horizon_n,
            "update_interval": update_interval,
            "replay_size": replay_size,
            "batch_size": batch_size,
            "greedy_epsilon": greedy_epsilon
        })
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        if sampler is None:
            sampler = sampling.TransitionSampler(MapPlayback(replay_size), batch_size, update_interval)
        kwargs.update({"sampler": sampler})
        self._network_optimizer = network_optimizer
        self._dim_state, self._dim_action = dim_state, dim_action
        self._f_model, self._sample_n, self._horizon_n = f_model, sample_n, horizon_n
        self._greedy_epsilon = greedy_epsilon
        super(MPCAgent, self).__init__(**kwargs)
        network_optimizer.add_updater(ModelUpdater(self.network), name="model")
        network_optimizer.compile()
        self._policy = WrapEpsilonGreedy(
            MPCPolicy(
                NetworkFunction({"goal": self.network["goal"], "reward": self.network["reward"]})
            )
        , epsilon=greedy_epsilon, num_actions=dim_action, is_continuous=True)

    def init_network(self, *args, **kwargs):
        state = tf.placeholder(tf.float32, [None, self._dim_state])
        action = tf.placeholder(tf.float32, [None, self._dim_action])
        net = Network([state, action], self._f_model, var_scope="model")
        return net

    def act(self, state, **kwargs):
        logging.warning("act:state:%s", state)
        action = self._policy.act(state)
        logging.warning("act:action:%s", action)
        return action

    def update_on_transition(self, batch):
        self._network_optimizer.update("model", self.sess, batch)
        info = self._network_optimizer.optimize_step(self.sess)
        return info, {}


