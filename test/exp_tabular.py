#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import hobotrl as hrl
from hobotrl.experiment import Experiment
from hobotrl.algorithms import tabular_q
import gym
import gym.spaces


class TabularGrid(Experiment):
    def run(self, args):
        env = hrl.envs.GridworldSink()

        agent = tabular_q.TabularQLearning(
            # TablularQMixin params
            num_action=env.action_space.n,
            discount_factor=0.9,
            # EpsilonGreedyPolicyMixin params
            epsilon_greedy=0.2
        )
        runner = hrl.envs.EnvRunner(env, agent)
        runner.episode(100)
Experiment.register(TabularGrid, "grid world with tabular-q learning")


class PendulumEnvWrapper(hrl.envs.C2DEnvWrapper):
    """
    wraps continuous state into discrete space
    """

    def __init__(self, env, quant_list=None, d2c_proc=None, action_n=None):
        super(PendulumEnvWrapper, self).__init__(env, quant_list, d2c_proc, action_n)
        self.observation_space = gym.spaces.Box(low=0, high=1024, shape=(1,))

    def _step(self, *args, **kwargs):
        next_state, reward, done, info = super(PendulumEnvWrapper, self)._step(*args, **kwargs)
        return [self.state_c2d(next_state)], reward, done, info

    def state_c2d(self, state):
        discrete_state = int((state[0] + 1.0) * 2), int((state[1]+1.0) * 2), int((state[2]+8.0) * 0.25)
        discrete_state = discrete_state[0] + (discrete_state[1] << 2) + (discrete_state[2] << 4)
        return discrete_state

    def _reset(self):
        return [self.state_c2d(super(PendulumEnvWrapper, self)._reset())]


class TabularPendulum(Experiment):
    """
    not converge yet
    """
    def run(self, args):
        action_n = 3
        reward_decay = 0.9
        env = gym.make("Pendulum-v0")
        env = PendulumEnvWrapper(env, [action_n])
        agent = tabular_q.TabularQLearning(
            # TablularQMixin params
            num_action=action_n,
            discount_factor=reward_decay,
            # EpsilonGreedyPolicyMixin params
            epsilon_greedy=0.2
        )
        runner = hrl.envs.EnvRunner(env, agent, reward_decay, evaluate_interval=100, render_interval=1000)
        runner.episode(20000)
Experiment.register(TabularPendulum, "Pendulum with tabular-q learning")


class TabularLake(Experiment):
    """
    """
    def run(self, args):
        reward_decay = 0.9
        env = gym.make("FrozenLake-v0")
        agent = tabular_q.TabularQLearning(
            # TablularQMixin params
            actions=range(env.action_space.n),
            gamma=reward_decay,
            # EpsilonGreedyPolicyMixin params
            epsilon=0.2
        )
        runner = hrl.envs.EnvRunner(env, agent, reward_decay, evaluate_interval=1000, render_interval=1000)
        runner.episode(20000)
Experiment.register(TabularLake, "FrozenLake with tabular-q learning")


if __name__ == '__main__':
    Experiment.main()
