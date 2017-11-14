#
# -*- coding: utf-8 -*-


import sys
sys.path.append(".")

from exp_algorithms import *
from car import *
from hobotrl.environments import *
from exp_car import *


def wrap_car(env, steer_n, speed_n):
    """Apply a common set of wrappers for Atari games."""
    env = CarDiscreteWrapper(env, steer_n, speed_n)
    env = MaxAndSkipEnv(env, skip=2, max_len=1)
    # env = ProcessFrame96H(env)
    env = FrameStack(env, 4)
    env = ScaledRewards(env, 0.1)
    env = ScaledFloatFrame(env)
    return env


class A3CToyCarDiscrete(A3CCarDiscrete2):
    def __init__(self, env=None, f_create_net=None, episode_n=10000, learning_rate=5e-5, discount_factor=0.95,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-1, 5e-3),
                 batch_size=32):
        if env is None:
            env = hrl.envs.ToyCarEnv()
            env = wrap_car(env, 5, 5)
            # gym.wrappers.Monitor(env, "./log/video", video_callable=lambda idx: True, force=True)
        super(A3CToyCarDiscrete, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                              batch_size)
Experiment.register(A3CToyCarDiscrete, "Discrete A3C for Toy Car Env")


class A3CToyCarContinuous(A3CCarContinuous):
    def __init__(self, env=None, f_create_net=None, episode_n=10000, learning_rate=5e-5, discount_factor=0.95,
                 entropy=hrl.utils.CappedLinear(1e6, 2e-4, 5e-5),
                 batch_size=32):
        if env is None:
            env = hrl.envs.ToyCarEnv()
            env = CarContinuousWrapper(env)
            env = MaxAndSkipEnv(env, skip=2, max_len=1)
            # env = ProcessFrame96H(env)
            env = FrameStack(env, 4)
            env = ScaledRewards(env, 0.1)
            env = ScaledFloatFrame(env)
        super(A3CToyCarContinuous, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                              batch_size)
Experiment.register(A3CToyCarContinuous, "Continuous A3C for Toy Car Env")


