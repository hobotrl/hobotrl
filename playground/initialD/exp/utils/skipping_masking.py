# -*- coding: utf-8 -*-
import logging
import numpy as np
import wrapt


class SkippingAgent(wrapt.ObjectProxy):
    def __init__(self, agent, n_skip, specific_act, *args, **kwargs):
        super(SkippingAgent, self).__init__(agent)
        self.n_skip = n_skip
        self.cnt_skip = n_skip - 1
        self._specific_act = specific_act
        self._state = None
        self._action = None
        self._reward = 0.0

    def set_n_skip(self, n_skip):
        self.n_skip = n_skip

    def act(self, state, **kwargs):
        if self.cnt_skip == self.n_skip - 1:
            action = self.__wrapped__.act(state, **kwargs)
        elif self._specific_act is None:
            action = self._action
        else:
            action = self._specific_act
        return action

    def step(self, state, action, reward, next_state, episode_done,
             *args, **kwargs):
        # keep the state before skipping
        if self.cnt_skip == self.n_skip - 1 or self._state is None:
            self._state = state
            self._action = action
        self._reward += reward
        info = {}
        if self.cnt_skip == 0 or episode_done:
            self._reward /= self.n_skip - self.cnt_skip
            print "Mean skip reward: {}".format(self._reward)
            info = self.__wrapped__.step(
                self._state, self._action, self._reward, next_state, episode_done,
                *args, **kwargs
            )
            self._reward = 0.0
            self.cnt_skip = self.n_skip
        self.cnt_skip -= 1
        return info


class RandFirstSkip(SkippingAgent):
    def __init__(self, *args, **kwargs):
        super(RandFirstSkip, self).__init__(*args, **kwargs)
        self.__base_n_skip = self.n_skip
        self._rand_start()

    def step(self, state, action, reward, next_state, episode_done,
             *args, **kwargs):
        if self.cnt_skip == 0:
            self.n_skip = self.__base_n_skip
        info = super(RandFirstSkip, self).step(
            state, action, reward, next_state, episode_done,
            *args, **kwargs
        )
        if episode_done:
            self._rand_start()
        return info

    def _rand_start(self):
        self.n_skip = int(self.__base_n_skip * (1 + np.random.rand()))
        self.cnt_skip = self.n_skip - 1
        logging.warning(
            "[RandFirstSkip]: random skip for first step {}/{}".format(
                self.n_skip, self.__base_n_skip
            )
        )


class NonUniformSkip(SkippingAgent):
    def __init__(self, n_skip_vec, *args, **kwargs):
        super(NonUniformSkip, self).__init__(*args, **kwargs)
        self.__n_skip_vec = n_skip_vec
        self.__base_n_skip = self.n_skip
        self._rand_start()


    def step(self, state, action, reward, next_state, episode_done,
             *args, **kwargs):
        if action is not None and action != 3:
            self.n_skip = self.__n_skip_vec[action]
            self.cnt_skip = self.n_skip - 1
        info = super(NonUniformSkip, self).step(
            state, action, reward, next_state, episode_done,
            *args, **kwargs
        )
        return info

    def _rand_start(self):
        self.n_skip = int(self.__base_n_skip * (1 + np.random.rand()))
        self.cnt_skip = self.n_skip - 1
        logging.warning(
            "[RandFirstSkip]: random skip for first step {}/{}".format(
                self.n_skip, self.__base_n_skip
            )
        )


class DynamicSkipping(SkippingAgent):
    def __init__(self, time_scales, *args, **kwargs):
        super(DynamicSkipping, self).__init__(*args, **kwargs)
        self._original_action_num = 3
        self.__base_n_skip = self.n_skip
        self._time_scales = time_scales
        assert self._specific_act == self._original_action_num * len(time_scales)
        self._n_step = 0
        self._rand_start()

    def step(self, state, action, reward, next_state, episode_done,
             *args, **kwargs):
        if self._n_step != 0:
            if action is not None and action != self._specific_act:
                self.n_skip = self._time_scales[action / self._original_action_num]
                self.cnt_skip = self.n_skip - 1

        info = super(DynamicSkipping, self).step(
            state, action, reward, next_state, episode_done,
            *args, **kwargs
        )
        self._n_step += 1
        if episode_done:
            self._n_step = 0
            self._rand_start()
        return info

    def _rand_start(self):
        self.n_skip = int(self.__base_n_skip * (1 + np.random.rand()))
        self.cnt_skip = self.n_skip - 1
        logging.warning(
            "[RandFirstSkip]: random skip for first step {}/{}".format(
                self.n_skip, self.__base_n_skip
            )
        )

# class MaskingAgent(Agent):
#     def __init__(self, mask_func, *args, **kwargs):
#         super(MaskingAgent, self).__init__(*args, **kwargs)
#         self._mask_func = mask_func
#
#     def act(self, state, **kwargs):
#         action = self._agent.act(state, **kwargs)
#         return self._mask_func(action, kwargs['condition'])





