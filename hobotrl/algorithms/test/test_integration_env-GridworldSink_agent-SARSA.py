# -*- coding: utf-8 -*-

"""Intergration test for tabular Q learning (mixin-based)

TODO: wrap up with Experiement?
"""

import sys
sys.path.append('../../../')

from hobotrl.algorithms.tabular_sarsa import SARSA
from hobotrl.environments import GridworldSink


env = GridworldSink()

agent = SARSA(
    # TablularQMixin params
    actions=env.ACTIONS,
    gamma=0.9,
    # EpsilonGreedyPolicyMixin params
    epsilon=0.02
)

while True:
    state, action, reward, next_state = None, None, 0.0, env.reset()
    done = False
    info = None
    cum_reward = 0.0
    n_steps = 0
    while True:
        next_action, update_info = agent.step(
            state=state, action=action,
            reward=reward, next_state=next_state,
            episode_done=done
        )
        state, action = next_state, next_action
        next_state, reward, done, info = env.step(action)
        n_steps += 1
        cum_reward += reward
        if done is True:
            # step agent for the last step
            action, update_info = agent.step(
                state=state, action=action,
                reward=reward, next_state=next_state,
                episode_done=done
            )
            print "Episode done in {} steps, reward is {}".format(
                n_steps, cum_reward
            )
            n_steps = 0
            cum_reward = 0.0
            raw_input('Next episode?')
            break

