# -*- coding: utf-8 -*-
"""
Estimators calculating target state value or target action value.
"""


import logging
import numpy as np
from utils import NP
from network import NetworkFunction


class TargetEstimator(object):

    def __init__(self, discount_factor=0.99):
        super(TargetEstimator, self).__init__()
        self._discount_factor = discount_factor

    def estimate(self, state, action, reward, next_state, episode_done):
        """
        estimate target state value (or action-value) from a batch of data.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :return:
        """
        raise NotImplementedError()


class OneStepTD(TargetEstimator):
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
    def __init__(self, q_function, discount_factor=0.99):
        super(NStepTD, self).__init__(discount_factor)
        self._q = q_function

    def estimate(self, state, action, reward, next_state, episode_done):
        batch_size = len(state)

        R = np.zeros(shape=[batch_size], dtype=float)
        if episode_done[-1]:
            r = 0.0
        else:
            # calculate from q_function(next_state)
            r = self._q([next_state[-1]])[0]
            r = np.max(r, axis=-1)
        for i in range(batch_size):
            index = batch_size - i - 1
            if episode_done[index] != 0:
                logging.warning("Terminated!, i:%s, reward:%s, Vi:%s", index, reward[index], r)
                r = 0
            r = reward[index] + self._discount_factor * r
            R[index] = r
        return R


class GAENStep(TargetEstimator):
    """
    target value, based on generalized advantage estimator
    https://arxiv.org/abs/1506.02438
    """

    def __init__(self, q_function, discount_factor=0.99, lambda_decay=0.95):
        super(GAENStep, self).__init__(discount_factor)
        self._q, self._lambda_decay = q_function, lambda_decay

    def estimate(self, state, action, reward, next_state, episode_done):
        batch_size = len(state)
        states = [s for s in state]
        if not episode_done[-1]:
            states.append(next_state[-1])  # need last next_state
        state_values = self._q(np.asarray(states))
        state_values = np.max(state_values, axis=1)
        if episode_done[-1]:
            state_values = np.append(state_values, 0.0)
        # 1 step TD
        delta = state_values[1:] * self._discount_factor + reward - state_values[:-1]
        factor = (self._lambda_decay * self._discount_factor) ** np.arange(batch_size)
        advantage = [np.sum(factor * delta)] \
                    + [np.sum(factor[:-i] * delta[i:]) for i in range(1, batch_size)]
        target_value = np.asarray(advantage) + state_values[:-1]
        return target_value

