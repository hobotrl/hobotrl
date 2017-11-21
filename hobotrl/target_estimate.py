# -*- coding: utf-8 -*-
"""
Estimators for target values of action or state value functions.

Modules include:
    TargetEstimator, OneStepTD, DDQNOneStepTD, NStepTD, GAENStep
    ContinuousActionEstimator, OptimalityTighteningEstimator
"""


import sys
import logging
import numpy as np
from utils import NP
from network import NetworkFunction


class TargetEstimator(object):

    def __init__(self, discount_factor=0.99):
        super(TargetEstimator, self).__init__()
        self._discount_factor = discount_factor

    def estimate(self, state, action, reward, next_state, episode_done, **kwargs):
        """
        Estimate target state value (or action-value) from a batch of data.

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :return:
        """
        raise NotImplementedError()


class OneStepTD(TargetEstimator):
    # TODO: or more formally this should be name OneStepBootstrappedBackup?
    def __init__(self, q_function, discount_factor):
        super(OneStepTD, self).__init__(discount_factor)
        self._q_function = q_function

    def estimate(self, state, action, reward, next_state, episode_done):
        target_q_val = self._q_function(next_state)
        target_q_val = np.max(target_q_val, axis=1)
        target_q_val = reward + self._discount_factor * target_q_val * (1.0 - episode_done)
        return target_q_val


class DDQNOneStepTD(TargetEstimator):
    def __init__(self, q_function, target_q_function, discount_factor=0.99):
        """

        :param q_function:
        :type q_function: NetworkFunction
        :param target_q_function:
        :type target_q_function: NetworkFunction
        :param discount_factor:
        """
        super(DDQNOneStepTD, self).__init__(discount_factor)
        self._q, self._target_q = q_function, target_q_function
        self._num_actions = q_function.output().op.shape.as_list()[-1]

    def estimate(self, state, action, reward, next_state, episode_done):
        learn_q_val = self._q(next_state)
        target_action = np.argmax(learn_q_val, axis=1)
        target_q_val = np.sum(self._target_q(next_state) * NP.one_hot(target_action, self._num_actions), axis=1)
        target_q_val = reward + self._discount_factor * target_q_val * (1.0 - episode_done)
        return target_q_val


class NStepTD(TargetEstimator):
    def __init__(self, v_function, discount_factor=0.99, bonus=None):
        """
        :param bonus: if it is a network.Function, it can output a intrinsic reward with proper inputs
        """
        self._bonus = bonus
        if self._bonus is not None:
            self.intrinsic_reward = 0.0
        super(NStepTD, self).__init__(discount_factor)
        self._v = v_function

    def estimate(self, state, action, reward, next_state, episode_done):
        batch_size = len(state)

        R = np.zeros(shape=[batch_size], dtype=float)

        if self._bonus:
            self.intrinsic_reward = self._bonus(state, next_state, action)
            reward += self.intrinsic_reward

        if episode_done[-1]:
            r = 0.0
        else:
            # calculate from q_function(next_state)
            r = self._v([next_state[-1]])[0]

        for i in range(batch_size):
            index = batch_size - i - 1
            r = reward[index] + self._discount_factor * r * (1.0 - episode_done[index])

            R[index] = r
        return R


class GAENStep(TargetEstimator):
    """
    target value, based on generalized advantage estimator
    https://arxiv.org/abs/1506.02438
    """

    def __init__(self, v_function, discount_factor=0.99, lambda_decay=0.95):
        super(GAENStep, self).__init__(discount_factor)
        self._v, self._lambda_decay = v_function, lambda_decay

    def estimate(self, state, action, reward, next_state, episode_done, **kwargs):
        batch_size = len(state)
        states = [s for s in state]
        if not episode_done[-1]:
            states.append(next_state[-1])  # need last next_state
        state_values = self._v(np.asarray(states))
        if episode_done[-1]:
            state_values = np.append(state_values, 0.0)
        # 1 step TD
        delta = state_values[1:] * self._discount_factor + reward - state_values[:-1]
        factor = (self._lambda_decay * self._discount_factor) ** np.arange(batch_size)
        advantage = [np.sum(factor * delta)] \
                    + [np.sum(factor[:-i] * delta[i:]) for i in range(1, batch_size)]
        target_value = np.asarray(advantage) + state_values[:-1]
        return target_value


class ContinuousActionEstimator(TargetEstimator):
    def __init__(self, v, discount_factor):
        super(ContinuousActionEstimator, self).__init__(discount_factor)
        self._v = v
        # self._actor, self._critic, = actor, critic

    def estimate(self, state, action, reward, next_state, episode_done, **kwargs):
        target_v = self._v(next_state)
        target_q = reward + self._discount_factor * (1.0 - episode_done) * target_v
        return target_q


class OptimalityTighteningEstimator(TargetEstimator):
    """
    Estimate target value to fit Q function according to:
    Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening
    https://arxiv.org/abs/1611.01606
    """
    def __init__(self, v_function, weight_upper=4.0, weight_lower=4.0, discount_factor=0.99):
        super(OptimalityTighteningEstimator, self).__init__(discount_factor)
        self._v = v_function
        self._weight_upper, self._weight_lower = weight_upper, weight_lower

    def estimate(self, state, action, reward, next_state, episode_done, **kwargs):
        """
        estimate target value for all states, within this trajectory
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :return:
        """
        states = np.concatenate((state, next_state[-1:]))
        state_value = self._v(states)
        td_target = reward + self._discount_factor * state_value[1:] * (1.0 - episode_done)
        upper_bounds, lower_bounds = 1.0 * td_target, 1.0 * td_target  # copy td_target as init
        for i in range(len(td_target)-2, -1, -1):
            l = max(lower_bounds[i+1], td_target[i+1]) * self._discount_factor + reward[i+1]
            lower_bounds[i] = max(l, lower_bounds[i])

        for i in range(1, len(td_target)):
            u = min(upper_bounds[i-1], td_target[i-1]) / self._discount_factor - reward[i-1] / self._discount_factor
            upper_bounds[i] = min(u, upper_bounds[i])
        w_all = 1.0 + self._weight_lower + self._weight_upper
        target_value = (td_target + self._weight_lower * lower_bounds + self._weight_upper * upper_bounds) / w_all
        return target_value
