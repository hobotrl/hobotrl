import sys
sys.path.append(".")
import logging
import numpy as np
import gym
import cv2
import matplotlib.colors as colors
from exp_algorithms import *
import hobotrl.environments as envs


class RewardSparsePendulum(gym.Wrapper):

    def __init__(self, env):
        super(RewardSparsePendulum, self).__init__(env)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        pass