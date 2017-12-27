# -*- coding: utf-8 -*-
"""helper wrapper classes."""
import time
import logging
import numpy as np
import wrapt

class EnvRewardVec2Scalar(wrapt.ObjectProxy):
    def __init__(self, env):
        super(EnvRewardVec2Scalar, self).__init__(env)
        self.reward_range = (-np.inf, np.inf)

        self._ema_speed = 10.0
        self._ema_dist = 0.0
        self._obs_risk = 0.0
        self._road_change = False
        self._mom_opp = 0.0
        self._mom_biking = 0.0
        self._steering = False

    def reset(self):
        state = self.__wrapped__.reset()

        self._ema_speed = 10.0
        self._ema_dist = 0.0
        self._obs_risk = 0.0
        self._road_change = False
        self._mom_opp = 0.0
        self._mom_biking = 0.0
        self._steering = False

        return state

    def step(self, action):
        next_state, rewards, done, info = self.__wrapped__.step(action)
        reward, info_diff  = self._func_scalar_reward(rewards, action)
        info.update(info_diff)
        early_done, info_diff = self._func_early_stopping()
        done = done | early_done
        info.update(info_diff)
        return next_state, reward, done, info

    def _func_scalar_reward(self, rewards, action):
        """Coverts a vector reward into a scalar."""
        info = {}

        # append a reward that is 1 when action is lane switching
        rewards = rewards.tolist()
        logging.warning((' '*3 + 'R: [' + '{:4.2f} ' * len(rewards) + ']').format(
            *rewards)),

        # extract relevant rewards.
        speed = rewards[0]
        dist = rewards[1]
        obs_risk = rewards[2]
        # road_invalid = rewards[3] > 0.01  # any yellow or red
        road_change = rewards[4] > 0.01  # entering intersection
        opp = rewards[5]
        biking = rewards[6]
        # inner = rewards[7]
        # outter = rewards[8]
        steer = np.logical_or(action == 1, action == 2)

        # update reward-related state vars
        ema_speed = 0.5 * self._ema_speed + 0.5 * speed
        ema_dist = 1.0 if dist > 2.0 else 0.9 * self._ema_dist
        mom_opp = min((opp < 0.5) * (self._mom_opp + 1), 20)
        mom_biking = min((biking > 0.5) * (self._mom_biking + 1), 12)
        steering = steer if action != 3 else self._steering
        self._ema_speed = ema_speed
        self._ema_dist = ema_dist
        self._obs_risk = obs_risk
        self._road_change = road_change
        self._mom_opp = mom_opp
        self._mom_biking = mom_biking
        self._steering = steering
        logging.warning('{:3.0f}, {:3.0f}, {:4.2f}, {:3.0f}'.format(
            mom_opp, mom_biking, ema_dist, self._steering)),

        # calculate scalar reward
        reward = [
            # velocity
            speed * 10 - 10,
            # obs factor
            -100.0 * obs_risk,
            # opposite
            -20 * (0.9 + 0.1 * mom_opp) * (mom_opp > 1.0),
            # ped
            -40 * (0.9 + 0.1 * mom_biking) * (mom_biking > 1.0),
            # steer
            steering * -40.0,
        ]
        reward = np.sum(reward) / 100.0
        logging.warning(': {:5.2f}'.format(reward))

        return reward, info

    def _func_early_stopping(self):
        """Several early stopping criterion."""
        info = {}
        done = False
        # switched lane while going into intersection.
        if self._road_change and self._ema_dist > 0.2:
            logging.warning("[Episode early stopping] turned into intersection.")
            done = True
            info['banned_road_change'] = True

        # used biking lane to cross intersection
        if self._road_change and self._mom_biking > 0:
            logging.warning("[Episode early stopping] entered intersection on biking lane.")
            done = True
            info['banned_road_change'] = True

        # hit obstacle
        if self._obs_risk > 1.0:
            logging.warning("[Episode early stopping] hit obstacle.")
            done = True

        return done, info


class EnvNoOpSkipping(wrapt.ObjectProxy):
    def __init__(self, env, n_skip, gamma, if_random_phase=True):
        super(EnvNoOpSkipping, self).__init__(env)
        self.__n_skip = n_skip
        self.__gamma = gamma
        self.__cnt_skip = None
        self.__if_random_phase = if_random_phase

    def reset(self):
        state = self.__wrapped__.reset()
        if self.__if_random_phase:
            self.__cnt_skip = int(self.__n_skip * (1 + np.random.rand()))
        else:
            self.__cnt_skip = self.__n_skip
        return state

    def step(self, action):
        next_state = None
        done = False
        info = {}
        skip_action = action
        n_skip = 0
        total_skip_reward = 0.0
        t = time.time()
        while not done and self.__cnt_skip > 0:
            next_state, reward, done, info = \
                self.__wrapped__.step(skip_action)
            skip_action = 3
            total_skip_reward += reward
            n_skip += 1
            self.__cnt_skip -= 1

        # average per-step reward
        total_skip_reward /= n_skip
        # penalize early stopping due to banned road change
        if 'banned_road_change' in info:
            total_skip_reward -= 1.0
        # assume last state is an absorbing state, compensate the reward
        # from this infinite series
        if done:
            total_skip_reward /= (1 - self.__gamma)
        # update info
        info['t_step'] = time.time() - t
        if done:
            info['flag_success'] = total_skip_reward > 0.0
        self.__cnt_skip = self.__n_skip
        logging.warning(
            'Mean skip reward: {:5.2f}'.format(total_skip_reward)
        )
        return next_state, total_skip_reward, done, info

    @property
    def n_skip(self):
        return self.__n_skip

    @n_skip.setter
    def n_skip(self, n_skip):
        self.__n_skip = n_skip