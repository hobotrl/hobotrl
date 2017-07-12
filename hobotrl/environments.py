#
# -*- coding: utf-8 -*-

import logging
from collections import deque
import tensorflow as tf
import gym
import numpy as np
import cv2


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
        action = self.agent.act(self.state, evaluate=evaluate)
        observation, reward, done, info = self.env.step(action)
        self.total_reward = reward + self.reward_decay * self.total_reward
        _, info = self.agent.step(state=self.state, action=action, reward=reward, next_state=observation,
                                  episode_done=done)
        self.record(info)
        self.state = observation
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


class AugmentEnvWrapper(gym.Wrapper):
    """
    wraps an environment, adding augmentation for reward/state/action.
    """

    def __init__(self, env,
                 reward_decay=0.99, reward_offset=0.0, reward_scale=1.0,
                 reward_shaping_proc=None,
                 state_offset=0, state_scale=1,
                 state_stack_n=None, state_stack_axis=-1,
                 state_skip=None,
                 state_augment_proc=None,
                 action_limit=None):
        """

        :param env:
        :param reward_decay:

        :param reward_offset:
        :param reward_scale: augmentation: reward = (reward + reward_offset) * reward_scale
        :param reward_shaping_proc: function processing rewards

        :param state_offset:
        :param state_scale: augmentation : state = (state + state_offset) * state_scale

        :param state_stack_n:
        :param state_stack_axis: if state_stack_n is not None, then multiple frames of state are stacked into
                    one state variable along state_stack_axis before returned by step().
                    The same action is repeated between these frames.

        :param action_limit: lower and upper limit of action for continuous action.
                    Assumes action accepted by step() would always range in [-1.0, 1.0],
                    and transformed into range [lower_limit, upper_limit] before applied to underlying env.

        :type action_limit: list
        """
        super(AugmentEnvWrapper, self).__init__(env)
        self.reward_decay, self.reward_offset, self.reward_scale = reward_decay, reward_offset, reward_scale
        self.state_offset, self.state_scale, self.stack_n, self.stack_axis, self.state_skip = \
            state_offset, state_scale, state_stack_n, state_stack_axis, state_skip
        self.state_augment_proc, self.reward_shaping_proc = state_augment_proc, reward_shaping_proc
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
            self.last_stacked_states = deque(maxlen=state_stack_n)  # lazy init
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

    def augment_reward(self, reward, observation=None, done=None, info=None):
        if self.reward_shaping_proc is not None:
            reward = self.reward_shaping_proc(reward, observation, done, info)
        return (reward + self.reward_offset) * self.reward_scale

    def pick_step(self, action):
        if self.state_skip is None or self.state_skip <= 0:
            observation, reward, done, info = self.env.step(action)
            reward = self.augment_reward(reward, observation, done, info)
            observation = self.augment_state(observation)
            return observation, reward, done, info
        total_reward = 0.0
        for i in range(self.state_skip):
            observation, reward, done, info = self.env.step(action)
            reward = self.augment_reward(reward, observation=observation, done=done, info=info)
            total_reward += reward
            if done:
                break
        return self.augment_state(observation), total_reward, done, info

    def _step(self, action):
        # augment action before apply
        action = self.augment_action(action)
        observation, reward, done, info = self.pick_step(action)
        if self.stack_n is not None:
            self.last_stacked_states.append(observation)
            observation = np.concatenate(self.last_stacked_states, self.stack_axis)
        return observation, reward, done, info

    def _reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        if self.stack_n is not None:
            for i in range(self.stack_n):
                self.last_stacked_states.append(state)
            state = np.concatenate(self.last_stacked_states, self.stack_axis)
        return state


class RewardShaping(object):
    def __init__(self):
        super(RewardShaping, self).__init__()
        pass

    def __call__(self, *args, **kwargs):
        pass


class InfoChange(RewardShaping):
    def __init__(self, increment_weight={}, decrement_weight={}):
        """
        reward shaping procedure for AugmentEnvWrapper:
            add additional reward according to changes in environments' info fields.
        :param increment_weight: map, weights of additional reward if fields in info increases.
        :param decrement_weight: map. weights of additional reward if fields in info decreases.
            a typical example for ALE env: {'ale.lives': -10} gives -10 when each life is lost.
        """
        self.increment_weight, self.decrement_weight = increment_weight, decrement_weight
        self.last_values = {}
        for k in increment_weight:
            self.last_values[k] = None
        for k in decrement_weight:
            self.last_values[k] = None
        super(InfoChange, self).__init__()

    def __call__(self, *args, **kwargs):
        reward = args[0]
        info = args[3]
        for k in self.last_values:
            if k in info:
                v = info.get(k)
                old_v = self.last_values[k]
                if k in self.increment_weight and old_v < v:
                    reward += (v - old_v) * self.increment_weight[k]
                elif k in self.decrement_weight and old_v > v:
                    reward += (old_v - v) * self.decrement_weight[k]
                self.last_values[k] = v
        return reward


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


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, max_len=2, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=max_len)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ClippedRewardsWrapper(gym.RewardWrapper):
    def _reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return (np.asarray(obs) + 0) * (1 / 255.0)
        # return np.array(obs).astype(np.float32) / 255.0


def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, max_len=1)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    # env = ClippedRewardsWrapper(env)
    return env
