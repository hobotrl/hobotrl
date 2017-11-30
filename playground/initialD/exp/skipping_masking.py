# -*- coding: utf-8 -*-

import time
import tensorflow as tf
from hobotrl.core import Agent
from hobotrl.sampling import TransitionSampler


class SkippingAgent(Agent):
    def __init__(self, agent, n_skip, specific_act, *args, **kwargs):
        super(SkippingAgent, self).__init__(*args, **kwargs)
        self._agent = agent
        self._n_skip = n_skip
        self._cnt_skip = n_skip
        self._specific_act = specific_act
        self._last_state = None


    def set_n_skip(self, n_skip):
        self._n_skip = n_skip

    def act(self, state, **kwargs):
        if self._cnt_skip == self._n_skip:
            action = self._agent.act(state, **kwargs)
            # maybe here exists a shallow copy problem
            self._last_state = state
            # print "self._last_state: ", self._last_state
        else:
            action = self._specific_act
        self._cnt_skip -= 1
        return action

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        if self._cnt_skip == 0 or episode_done:
            self._agent.step(self._last_state, action, reward, next_state, episode_done, **kwargs)
            self._cnt_skip = self._n_skip


class MaskingAgent(Agent):
    def __init__(self, mask_func, *args, **kwargs):
        super(MaskingAgent, self).__init__(*args, **kwargs)
        self._mask_func = mask_func

    def act(self, state, **kwargs):
        action = self._agent.act(state, **kwargs)
        return self._mask_func(action, kwargs['condition'])





