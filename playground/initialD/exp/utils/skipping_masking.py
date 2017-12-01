# -*- coding: utf-8 -*-

import time
import tensorflow as tf
from hobotrl.core import Agent
from hobotrl.sampling import TransitionSampler
import sys
import numpy as np

class SkippingAgent(Agent):
    def __init__(self, agent, n_skip, specific_act, *args, **kwargs):
        super(SkippingAgent, self).__init__(*args, **kwargs)
        self._agent = agent
        self._n_skip = n_skip
        self._cnt_skip = None
        self._specific_act = specific_act
        self._last_state = None
        self._reward = 0.0

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

    def create_session(self, config=None, save_dir=None, **kwargs):
        return self._agent.create_session(config=config,
                                          save_dir=save_dir, **kwargs)

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        self._reward += reward
        if self._cnt_skip == 0 or episode_done:
            self._reward /= self._n_skip - self._cnt_skip
            self._agent.step(self._last_state, action, self._reward, next_state, episode_done, **kwargs)
            self._cnt_skip = self._n_skip
            self._reward = 0.0


class RandFirstSkip(SkippingAgent):
    def __init__(self, agent, n_skip, specific_act, *args, **kwargs):
        super(RandFirstSkip, self).__init__(agent, n_skip, specific_act, *args, **kwargs)
        self._n_step = 0
        self._base_n_skip = self._n_skip

    def act(self, state, **kwargs):
        if self._n_step == 0:
            self.set_n_skip(int(self._base_n_skip * (1 + np.random.rand())))
            self._cnt_skip = self._n_skip

        if self._n_step == self._n_skip:
            self._n_skip = self._base_n_skip
            self._cnt_skip = self._n_skip

        return self._agent.act(state, **kwargs)


class AdjustSkippingAgent(SkippingAgent):
    def __init__(self, agent, n_skip, specific_act, n_skip_vec, *args, **kwargs):
        super(AdjustSkippingAgent, self).__init__(*args, **kwargs)
        self._n_skip_vec = n_skip_vec

    def act(self, state, **kwargs):
        action = self._agent.act(state, **kwargs)
        if self._cnt_skip == self._n_skip - 1:
            self.set_n_skip(self._n_skip_vec[action])
            self._cnt_skip = self._n_skip - 1
        return action


class MaskingAgent(Agent):
    def __init__(self, mask_func, *args, **kwargs):
        super(MaskingAgent, self).__init__(*args, **kwargs)
        self._mask_func = mask_func

    def act(self, state, **kwargs):
        action = self._agent.act(state, **kwargs)
        return self._mask_func(action, kwargs['condition'])





