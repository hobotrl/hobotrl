#
# -*- coding: utf-8 -*-
import sys
import gym
import gym.spaces
import numpy as np
import matplotlib.colors as colors
import logging

from hobotrl.environments.environments import *


class CarDiscreteWrapper(gym.Wrapper):
    """
    Wraps car env into discrete action control problem
    """

    def __init__(self, env, steer_n, speed_n):
        super(CarDiscreteWrapper, self).__init__(env)
        self.steer_n, self.speed_n = steer_n, speed_n
        self.env = env
        self.action_n = steer_n * speed_n
        self.action_space = gym.spaces.discrete.Discrete(self.action_n)

    def __getattr__(self, name):
        print("getattr:", name, " @ ", id(self.env))
        if name == "action_space":
            print("getattr: action_space:", name)
            return self.action_space
        else:
            return getattr(self.env, name)

    def _step(self, action):
        action_c = self.action_d2c(action)
        next_state, reward, done, info = self.env.step(action_c)

        return next_state, reward, done, info

    def action_c2d(self, action):
        """
        continuous action to discrete action
        :param action:
        :return:
        """
        steer_i = int((action[0] - (-1.0)) / 2.0 * self.steer_n)
        steer_i = self.steer_n - 1 if steer_i >= self.steer_n else steer_i
        if abs(action[1]) > abs(action[2]):
            speed_action = action[1]
        else:
            speed_action = -action[2]
        speed_i = int((speed_action - (-1.0)) / 2.0 * self.speed_n)
        speed_i = self.speed_n - 1 if speed_i >= self.speed_n else speed_i
        return steer_i * self.speed_n + speed_i

    def action_d2c(self, action):
        steer_i = int(action / self.speed_n)
        speed_i = action % self.speed_n
        action_c = np.asarray([0., 0., 0.])
        action_c[0] = float(steer_i) / self.steer_n * 2 - 1.0 + 1.0 / self.steer_n
        speed_c = float(speed_i) / self.speed_n * 2 - 1.0 + 1.0 / self.speed_n
        if speed_c >= 0:
            action_c[1], action_c[2] = speed_c, 0
        else:
            action_c[1], action_c[2] = 0, -speed_c
        return action_c


class CarContinuousWrapper(gym.Wrapper):

    def __init__(self, env):
        super(CarContinuousWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, [2])

    def _step(self, action):
        env_action = np.zeros(3)
        env_action[0] = action[0]
        if action[1] > 0:
            env_action[1], env_action[2] = action[1], 0
        else:
            env_action[1], env_action[2] = 0, -action[1]
        return self.env.step(env_action)


class CarGrassWrapper(gym.Wrapper):

    def __init__(self, env, grass_penalty=0.5):
        super(CarGrassWrapper, self).__init__(env)
        self.grass_penalty = grass_penalty

    def _step(self, action):

        ob, reward, done, info = self.env.step(action)
        if (ob[71:76, 47:49, 0] > 200).all():  # red car visible
            front = (ob[70, 47:49, 1] > 200).all()
            back = (ob[76, 47:49, 1] > 200).all()
            left = (ob[71:74, 46, 1] > 200).all()
            right = (ob[71:74, 49, 1] > 200).all()
            if front and back and left and right:
                reward -= self.grass_penalty
        return ob, reward, done, info


class ProcessFrame96H(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame96H, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 1))

    def _observation(self, obs):
        return ProcessFrame96H.process(obs)

    @staticmethod
    def process(frame):
        if list(frame.shape) == [96, 96, 3]:
            pass
        else:
            assert False, "Unknown resolution."
        img = frame
        img = colors.rgb_to_hsv(img / 255.0)
        img = np.transpose(img, axes=[2, 0, 1])[0]
        img = (img * 255).astype(np.uint8).reshape((96, 96, 1))
        return img


def wrap_car(env, steer_n, speed_n, skipFrame=2):
    """Apply a common set of wrappers for Atari games."""
    env = CarDiscreteWrapper(env, steer_n, speed_n)
    env = ScaledFloatFrame(env)
    env = MaxAndSkipEnv(env, skip=skipFrame, max_len=1)
    # env = ProcessFrame96H(env)
    env = FrameStack(env, 4)
    env = ScaledRewards(env, 0.1)
    # env = RemapFrame(env)
    #env = HalfFrame(env)
    return env


class CarEarlyTermWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CarEarlyTermWrapper, self).__init__(env)
        self._n_term = None
        self._n_step = None
        self._ep_total_reward = None

    def _step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self._ep_total_reward += reward
        self._n_step += 1
        if self._n_step >= self._n_term:
            done = True
            logging.warning(
                "CarEarlyTermWrapper._step(): "
                "early termination at step {}.".format(self._n_step)
            )
        if done:
            info['episode_step_reward_original'] = \
                self._ep_total_reward / self._n_step
        return next_state, reward, done, info

    def _reset(self, **kwargs):
        self._n_step = 0
        self._n_term = np.random.randint(1, 301)
        self._ep_total_reward = 0.0
        return self.env.reset(**kwargs)


class CarTailCompensationWrapper(gym.Wrapper):
    def __init__(self, env, discount_factor, if_compensate=True):
        super(CarTailCompensationWrapper, self).__init__(env)
        self._if_compensate = if_compensate
        self._gamma = discount_factor

    def _step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if done and self._if_compensate:
            reward /= (1 - self._gamma)
            logging.warning(
                "CarTailCompensationWrapper._step(): "
                "compensated reward {}".format(reward)
            )
        return next_state, reward, done, info