#
# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
import gym
import numpy as np


class EnvRunner(object):
    """
    interaction between agent and environment.
    """
    def __init__(self, env, agent, reward_decay=0.99, max_episode_len=5000, evaluate_interval=20, render_interval=1,
                 logdir=None):
        """

        :param env: environment.
        :param agent: agent.
        :param reward_decay: deprecated. EnvRunner should not discount future rewards.
        :param max_episode_len:
        :param evaluate_interval:
        :param render_interval:
        :param logdir: dir to save info from agent as tensorboard log.
        """
        super(EnvRunner, self).__init__()
        self.env, self.agent = env, agent
        self.reward_decay, self.max_episode_len = 1.0, max_episode_len
        self.evaluate_interval, self.render_interval = evaluate_interval, render_interval
        self.episode_n, self.step_n = 0, 0
        self.state = None
        self.action = None
        self.total_reward = 0.0
        self.summary_writer = None
        if logdir is not None:
            self.summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    def step(self, evaluate=False):
        """
        agent runs one step against env.
        :param evaluate:
        :return:
        """
        self.step_n += 1
        # TODO: directly calling agent.act will by-pass BaseDeepAgent, which
        # checks and assigns 'sess' arugment. So we manually set sess here. But
        # is there a better way to do this?
        self.action = self.agent.act(
            state=self.state, evaluate=evaluate, sess=self.agent.sess
        )
        next_state, reward, done, info = self.env.step(self.action)
        self.total_reward = reward + self.reward_decay * self.total_reward
        _, info = self.agent.step(
            state=self.state, action=self.action, reward=reward,
            next_state=next_state, episode_done=done
        )
        self.record(info)
        self.state = next_state
        return done

    def record(self, info):
        if self.summary_writer is not None:
            for name in info:
                value = info[name]
                summary = tf.Summary()
                summary.value.add(tag=name, simple_value=np.mean(value))
                self.summary_writer.add_summary(summary, self.step_n)

    def episode(self, n):
        """
        agent runs n episodes against env.
        :param n:
        :return:
        """
        for i in range(n):
            self.episode_n += 1
            self.state = self.env.reset()
            self.agent.new_episode(self.state)
            self.total_reward = 0.0
            evaluate = self.episode_n % self.evaluate_interval == 0
            render = self.episode_n % self.render_interval == 0
            if evaluate:
                logging.warning("Episode %d evaluating", self.episode_n)
            t = 0
            for t in range(self.max_episode_len):
                if render:
                    self.env.render()
                terminate = self.step(evaluate)
                if terminate:
                    break
            logging.warning("Episode %d finished after %d steps, total reward=%f", self.episode_n, t + 1,
                                self.total_reward)
            self.record({"episode_total_reward": self.total_reward})


class AugmentEnvWrapper(object):
    """
    wraps an environment, adding augmentation for reward/state/action.
    """

    def __init__(self, env,
                 reward_decay=0.99, reward_offset=0.0, reward_scale=1.0,
                 state_offset=0, state_scale=1,
                 state_stack_n=None, state_stack_axis=-1,
                 state_augment_proc=None,
                 action_limit=None):
        """

        :param env:
        :param reward_decay:
        :param reward_offset:
        :param reward_scale:
        :param state_offset:
        :param state_scale:
        :param state_stack_n:
        :param state_stack_axis:
        :param action_limit: lower and upper limit of action for continuous action
        :type action_limit: list
        """
        self.env = env
        self.reward_decay, self.reward_offset, self.reward_scale = reward_decay, reward_offset, reward_scale
        self.state_offset, self.state_scale, self.stack_n, self.stack_axis = \
            state_offset, state_scale, state_stack_n, state_stack_axis
        self.state_augment_proc = state_augment_proc
        self.is_continuous_action = env.action_space.__class__.__name__ == "Box"
        if self.is_continuous_action:
            self.action_limit = action_limit
            self.action_scale = (action_limit[1] - action_limit[0])/2.0
            self.action_offset = (action_limit[1] + action_limit[0])/2.0
            logging.warning("limit:%s, scale:%s, offset:%s", action_limit, self.action_scale, self.action_offset)
        if state_stack_n is not None:
            self.last_action, self.stack_counter, self.last_reward = None, 0, 0.0
            self.state_shape = None  # state dimension of 1 frame
            space = env.observation_space
            state_low, state_high, state_shape = space.low, space.high, list(space.shape)
            state_shape[self.stack_axis] *= self.stack_n
            # if type(state_low) == type(state_shape):
            #     # ndarray low and high
            #     state_low, state_high = [np.repeat(l, self.stack_n, axis=self.stack_axis)
            #                              for l in [state_low, state_high]]

            self.observation_space = gym.spaces.box.Box(np.min(state_low), np.max(state_high), state_shape)
            self.state_shape = state_shape
            self.last_stacked_states = []  # lazy init
            pass

    def __getattr__(self, name):
        if self.stack_n is not None and name == "observation_space":
            return self.observation_space
        return getattr(self.env, name)

    def augment_action(self, action):
        if self.is_continuous_action and self.action_limit is not None:
            action = (action + self.action_offset) * self.action_scale
        return action

    def augment_state(self, state):
        if self.state_augment_proc is not None:
            state = self.state_augment_proc(state)
        return (state + self.state_offset) * self.state_scale

    def augment_reward(self, reward):
        return (reward + self.reward_offset) * self.reward_scale

    def step(self, action):
        # augment action before apply
        action = self.augment_action(action)
        if self.stack_n is not None:
            stacked_reward = 0.0
            new_states = []
            for i in range(self.stack_n):
                observation, reward, done, info = self.env.step(action)
                stacked_reward = self.augment_reward(reward) + self.reward_decay * stacked_reward
                new_states.append(self.augment_state(observation))
                if done:
                    break
            if len(new_states) < self.stack_n:  # episode ends before stack_n
                new_states = self.last_stacked_states[-(self.stack_n - len(new_states)):] + new_states
            stacked_state = np.concatenate(new_states, self.stack_axis)
            observation, reward = stacked_state, stacked_reward
            self.last_stacked_states = new_states
        else:
            observation, reward, done, info = self.env.step(action)
            observation = self.augment_state(observation)
            reward = self.augment_reward(reward)
        # augment state/reward before return
        return observation, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        if self.stack_n is not None:
            self.last_stacked_states = [state for i in range(self.stack_n)]
            state = np.concatenate(self.last_stacked_states, self.stack_axis)
        return state


class C2DEnvWrapper(object):
    """
    wraps an continuous action env to discrete env.
    """

    def __init__(self, env, quant_list=None, d2c_proc=None, action_n=None):
        """
        either quant_list or d2c_proc is not None
        :param env: continuous action env
        :param quant_list: [2, 3, 4] if i want 2 value for action dimension 0, 3 value for dim 1, 4 for dim 2.
        :param d2c_proc: function converting discrete action to continuous
        :param action_n: count of discrete actions, if d2c_proc is not None
        """
        self.env = env
        if quant_list is not None:
            self.quant_list = quant_list
            self.action_n = reduce(lambda x, y: x * y, self.quant_list)
            self.action_space = gym.spaces.discrete.Discrete(self.action_n)
            self.action_table = []
            for low, high, n in zip(env.action_space.low, env.action_space.high, self.quant_list):
                self.action_table.append((np.arange(n)) * (high - low) / float(n-1) + low)
            logging.warning("action_table:%s", self.action_table)
            self.d2c_proc = lambda action: self.action_d2c(action)
        else:
            # use d2c_proc
            self.action_n = action_n
            self.action_space = gym.spaces.discrete.Discrete(self.action_n)
            self.d2c_proc = d2c_proc

    def __getattr__(self, name):
        if name == "action_space":
            print("getattr: action_space:", name)
            return self.action_space
        else:
            return getattr(self.env, name)

    def step(self, *args, **kwargs):
        # lives_before = self.env.ale.lives()
        if len(args) > 0:
            action_i = args[0]
        else:
            action_i = kwargs["action"]
        action_proc = self.d2c_proc
        action_c = action_proc(action_i)
        # logging.warning("action d2c: %s => %s", action_i, action_c)
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
        action_index = []
        for q in self.quant_list:
            action_index.append(action % q)
            action /= q
        return [self.action_table[i][x] for i, x in enumerate(action_index)]

    def reset(self):
        return self.env.reset()


class GridworldSink:
    """A simple maze game with a single goal state
    This is a simple maze game environment. The game is played on a 2-D
    grid world and one of the grids is the goal grid. The agent can move up,
    down, left, and right in order to reach the goal grid, and the episode
    will end when the agent arrives at the goal grid.

    The grids by the boundary are fenced. The agent can choose to move
    towards the fence, but will remain there and (optionally) receive a
    "wall_reward".
    """

    def __init__(self, dims=None, goal_state=None,
                 goal_reward=100, wall_reward=0, null_reward=0):
        """
        Paramters
        ---------
        dims :
        goal_state  :
        goal_reward :
        wall_reward :
        null_reward :
        """
        self.ACTIONS = ['left', 'right', 'up', 'down']  # legitimate ACTIONS

        if dims is None:
            self.DIMS = (4, 5)
        else:
            self.DIMS = dims

        if goal_state is None:
            self.GOAL_STATE = (2, 2)
        else:
            self.GOAL_STATE = goal_state

        self.GOAL_REWARD = goal_reward
        self.WALL_REWARD = wall_reward
        self.NULL_REWARD = null_reward

        self.state = None
        self.done = False

        self.reset()

    def step(self, action):
        """

        Parameters
        ----------
        action : must be contailed in self.ACTIONS
        -------

        """
        if self.done:
            raise ValueError("Episode done, please restart.")

        next_state, reward, self.done = self.transition_(self.state, action)
        self.state = next_state
        return next_state, reward, self.done, None

    def transition_(self, current_state, action):
        """State transition and rewarding logic

        """
        if action == 'up':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[0] == 0 \
                else ((current_state[0] - 1, current_state[1]), self.NULL_REWARD)
        elif action == 'down':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[0] == (self.DIMS[0] - 1) \
                else ((current_state[0] + 1, current_state[1]), self.NULL_REWARD)
        elif action == 'left':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[1] == 0 \
                else ((current_state[0], current_state[1] - 1), self.NULL_REWARD)
        elif action == 'right':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[1] == (self.DIMS[1] - 1) \
                else ((current_state[0], current_state[1] + 1), self.NULL_REWARD)
        else:
            print 'I don\'t understand this action ({}), I\'ll stay.'.format(action)
            next_state, reward = current_state, self.NULL_REWARD

        done = False
        if next_state == self.GOAL_STATE:
            reward = self.GOAL_REWARD
            done = True

        return next_state, reward, done

    def optimal_policy(self, state):
        if state[0] < self.GOAL_STATE[0]:
            return 'down'
        elif state[0] > self.GOAL_STATE[0]:
            return 'up'
        elif state[1] < self.GOAL_STATE[1]:
            return 'right'
        else:
            return 'left'

    def reset(self):
        """Randomly throw the agent to a non-goal state

        """
        next_state = self.GOAL_STATE
        while next_state == self.GOAL_STATE:
            next_state = (np.random.randint(0, self.DIMS[0]), np.random.randint(0, self.DIMS[1]))
        self.state = next_state
        self.done = False

        return self.state

    def isDone(self):
        return self.state == self.GOAL_STATE

    def render(self):
        # I can't render
        pass

