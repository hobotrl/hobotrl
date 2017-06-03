import sys
sys.path.append('../')

from mixin_style import SARSA
from simple_env import GridworldSink


env = GridworldSink()

agent = SARSA(
    # TablularQMixin params
    actions=env.ACTIONS,
    gamma=0.9,
    # EpsilonGreedyPolicyMixin params
    epsilon=0.02
)

while True:
    last_state, last_action, state, reward = None, None, env.reset(), 0.0
    done = False
    info = None
    cum_reward = 0.0
    n_steps = 0
    while True:
        action, update_info = agent.step(
            last_state=last_state, last_action=last_action,
            state=state, reward=reward, episode_done=done)
        last_state, last_action = state, action  # update exp buffer
        state, reward, done, info = env.step(action)
        n_steps += 1
        cum_reward += reward
        if done is True:
            # step agent for the last step
            action, update_info = agent.step(
                last_state=last_state, last_action=last_action,
                state=state, reward=reward, episode_done=done)
            print "Episode done in {} steps, reward is {}".format(
                n_steps, cum_reward
            )
            n_steps = 0
            cum_reward = 0.0
            raw_input()
            break

