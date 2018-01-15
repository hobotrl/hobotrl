# -*- coding: utf-8 -*-

"""Intergration test for tabular Q learning (mixin-based)

TODO: wrap up with Experiement?
"""

import sys
sys.path.append('../../../')

import numpy as np
from hobotrl.algorithms.tabular_q import TabularQLearning
from hobotrl.environments import GridworldSink


env = GridworldSink()

agent = TabularQLearning(
    num_action=env.action_space.n,
    discount_factor=0.9,
    epsilon_greedy=0.2
)

while True:
    state, action, reward, next_state = None, None, 0.0, env.reset()
    done = False
    info = None
    cum_reward = 0.0
    n_steps = 0
    while True:
        next_action = agent.act(next_state)
        state, action = next_state, next_action
        next_state, reward, done, info = env.step(action)
        n_steps += 1
        cum_reward += reward
        info = agent.step(state, action, reward, next_state, done)
        if done is True:
            print "Episode done in {} steps, reward is {}".format(
                n_steps, cum_reward
            )
            n_steps = 0
            cum_reward = 0.0
            raw_input('Next episode?')
            break

