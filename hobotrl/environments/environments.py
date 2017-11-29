#
# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
import cv2
from collections import deque


class EnvRunner(object):
    """
    interaction between agent and environment.
    """
    def __init__(self, env, agent, reward_decay=0.99, max_episode_len=5000,
                 evaluate_interval=sys.maxint, render_interval=sys.maxint,
                 render_once=False,
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
            self.summary_writer = SummaryWriterCache.get(logdir)
        self.render_once = True if render_once else False

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
        info = self.agent.step(
            state=self.state, action=self.action, reward=reward,
            next_state=next_state, episode_done=done
        )
        self.record(info)
        self.state = next_state
        if self.render_once:
            self.env.render()
            self.render_once = False
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
        rewards = []
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
            rewards.append(self.total_reward)
        return rewards


class EnvRunner2(object):
    def __init__(self, env, agent,
                 n_episodes=-1, moving_average_window_size=50,
                 no_reward_reset_interval=-1, no_reward_punishment=0,
                 checkpoint_save_interval=-1, log_dir=None, log_file_name=None,
                 render_env=False, render_interval=1, render_length=200, frame_time=0, render_options={},
                 show_frame_rate=False, show_frame_rate_interval=100):
        """
        Trains the agent in the environment.

        :param env: the environment.
        :param agent: the agent.
        :param n_episodes: number of episodes before terminating, -1 means run forever.
        :param moving_average_window_size: window size for calculating moving average of rewards.
        :param no_reward_reset_interval: reset after this number of steps if no reward is received.
        :param no_reward_punishment: punishment when being reset for no reward.
        :param checkpoint_save_interval: save checkpoint every this number of steps.
        :param log_dir: path to save log files.
        :param log_file_name: file name of the csv file.
        :param render_env: whether to render the environment during the training.
        :param render_interval: number of steps between each render session.
        :param render_length: length of render session, counted in steps.
        :param frame_time: time interval between each frame, in seconds.
        :param render_options: options that will pass to env.render().
        """
        assert n_episodes >= -1
        assert moving_average_window_size >= 1
        assert no_reward_reset_interval >= -1
        assert checkpoint_save_interval >= -1
        assert render_interval >=1
        assert render_length > 0
        assert frame_time >= 0

        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.moving_average_window_size = moving_average_window_size
        self.no_reward_reset_interval = no_reward_reset_interval
        self.no_reward_punishment = no_reward_punishment
        self.checkpoint_save_interval = checkpoint_save_interval
        self.log_dir = log_dir
        self.render_env = render_env
        self.render_interval = render_interval
        self.render_length = render_length
        self.frame_time = frame_time
        self.render_options = render_options
        self.show_frame_rate = show_frame_rate
        self.show_frame_rate_interval = show_frame_rate_interval

        self.episode_count = 0  # Count episodes
        self.step_count = 0  # Count number of total steps
        self.current_episode_reward = 0.
        self.reward_history = list()  # Record the total reward of last a few episodes
        self.last_reward_step = 0  # The step when the agent gets last reward
        self.current_episode_step_count = 0  # The step count of current episode
        self.loss_sum = 0.  # Sum of loss in current episode

        # Open log file
        if log_file_name:
            assert log_dir
            self.log_file = open(os.path.join(log_dir, log_file_name), "w")
        else:
            self.log_file = None

        # Open summary writer
        if log_dir:
            self.summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        else:
            self.summary_writer = None

    def run(self):
        """
        Start training.
        """
        # Initialize environment
        state = self.env.reset()

        last_frame_rate_check_time = time.time()  # Time when last frame rate check was done

        while self.episode_count != self.n_episodes:
            # Run a step
            state, done = self.step(state)

            # Render if needed
            if self.render_env and self.step_count % self.render_interval <= self.render_length:
                render_result = self.env.render(**self.render_options)

                # Print to terminal if the render result is a string(e.g. when render mode is "ansi")
                if render_result:
                    print render_result

                time.sleep(self.frame_time)

                # Close the window at the last frame
                if self.step_count % self.render_interval == self.render_length:
                    self.env.render(close=True)

            # Save data to log files at the end of each episode
            if done:
                # Save to csv
                if self.log_file:
                    print "Episode %d Step %d:" % (self.episode_count, self.step_count),
                    print "%7.2f/%.2f" % (self.reward_history[-1], self.reward_summary)

                    self.log_file.write("%d,%d,%f,%f\n" % (self.episode_count, self.step_count, self.reward_history[-1], self.reward_summary))

                # Save to summary writer
                if self.summary_writer:
                    summary = tf.Summary()
                    summary.value.add(tag="step count", simple_value=self.step_count)
                    summary.value.add(tag="reward", simple_value=self.reward_history[-1])
                    summary.value.add(tag="average reward", simple_value=self.reward_summary)
                    if str(self.loss_summary) != 'nan':
                        summary.value.add(tag="loss", simple_value=self.loss_summary)
                    else:
                        summary.value.add(tag="loss", simple_value=0)

                    self.summary_writer.add_summary(summary, self.episode_count)

            # Save checkpoint if needed
            if self.checkpoint_save_interval != -1 and self.step_count % self.checkpoint_save_interval == 0:
                saver = tf.train.Saver()
                saver.save(self.agent.get_session(), os.path.join(self.log_dir, '%d.ckpt' % self.step_count))
                print "Checkpoint saved at step %d" % self.step_count

            # Count steps
            self.step_count += 1
            if done:
                self.current_episode_step_count = 0
                self.loss_sum = 0.
            else:
                self.current_episode_step_count += 1

            # Calculate frame rate if needed
            if self.show_frame_rate and self.step_count % self.show_frame_rate_interval == 0:
                print "Frame rate:", self.show_frame_rate_interval/(time.time() - last_frame_rate_check_time)
                last_frame_rate_check_time = time.time()

    def step(self, state):
        """
        Take a step.

        :param state: current state
        :return: a tuple: (next state, whether current episode is done)
        """
        # Take a step
        action = self.agent.act(state, show_action_values=self.render_env)
        next_state, reward, done, info = self.env.step(action)

        # Reset if no reward is seen for last a few steps
        if reward > 1e-6:
            self.last_reward_step = self.step_count

        if self.step_count - self.last_reward_step == self.no_reward_reset_interval:
            print "Reset for no reward"
            done = True
            reward -= self.no_reward_punishment

        # Train the agent
        info = self.agent.reinforce_(state=state,
                                     action=action,
                                     reward=reward,
                                     next_state=next_state,
                                     episode_done=done)

        try:
            loss = info["loss"]
        except KeyError:
            loss = float("nan")

        # Print reward if needed
        if self.render_env and abs(reward) > 1e-6:
            print "%.1f" % reward

        # Record reward and loss
        self.current_episode_reward += reward
        self.loss_sum += loss
        # Episode done
        if done:
            next_state = self.env.reset()

            self.save_reward_record()
            self.last_reward_step = self.step_count

            self.episode_count += 1

        return next_state, done

    def save_reward_record(self):
        """
        Save reward record for current episode.
        """
        self.reward_history.append(self.current_episode_reward)
        self.current_episode_reward = 0.

        # Trim the history record if it's length is longer than moving_average_window_size
        if len(self.reward_history) > self.moving_average_window_size:
            del self.reward_history[0]

    @property
    def reward_summary(self):
        """
        Get the average reward of last few episodes.
        """
        return float(sum(self.reward_history))/len(self.reward_history)

    @ property
    def loss_summary(self):
        try:
            return self.loss_sum/self.current_episode_step_count
        except ZeroDivisionError:
            return None

    def load_checkpoint(self, file_path, step_count=None):
        """
        Load a checkpoint.

        :param file_path: path to the checkpoint
        :param step_count: start to count steps with this number
        :return:
        """
        saver = tf.train.Saver()
        saver.restore(self.agent.get_session(), os.path.join(self.log_dir, file_path))

        if step_count:
            self.step_count = step_count

    def run_demo(self, file_name, show_action_values=False, pause_before_start=True):
        """
        Load a checkpoint and show a demo.

        :param file_name: the checkpoint's file name.
        :param show_action_values: whether to show action values on terminal.
        """
        def render():
            render_result = self.env.render(**self.render_options)
            if render_result:
                print render_result
            time.sleep(self.frame_time)

        self.load_checkpoint(file_name)

        # Render first frame
        state = self.env.reset()
        render()
        if pause_before_start:
            raw_input("Press Enter to start demonstration")

        while True:
            # Act
            action = self.agent.act(state, show_action_values=show_action_values)
            state, reward, done, info = self.env.step(action)

            # Render
            render()

            # Reset if episode ends
            if done:
                self.env.reset()
                render()
                if pause_before_start:
                    raw_input("Press Enter to start demonstration")

class AugmentEnvWrapper(gym.Wrapper):
    """
    wraps an environment, adding augmentation for reward/state/action.
    """

    def __init__(self, env,
                 reward_decay=0.99, reward_offset=0.0, reward_scale=1.0,
                 reward_shaping_proc=None,
                 state_offset=0, state_scale=1,
                 state_stack_n=1, state_stack_axis=-1,
                 state_skip=1,
                 state_augment_proc=None,
                 action_limit=None,
                 amend_reward_decay=False,
                 random_start=False,
                 discard_skipped_frames=True
                 ):
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

        :param state_skip: number of frames for frame skip. If not specified, this value will be set to state_stack_n.

        :param state_augment_proc:

        :param action_limit: lower and upper limit of action for continuous action.
                    Assumes action accepted by step() would always range in [-1.0, 1.0],
                    and transformed into range [lower_limit, upper_limit] before applied to underlying env.

        :type action_limit: list
        :param amend_reward_decay: whether to amend reward decay when state_stack_n is not None.
        :param random_start: take random actions at the beginning of each episode to fill the stack queue
        :param discard_skipped_frames: if set to True, frames that are skipped won't be stacked.
        """
        assert 0. <= reward_decay <= 1.
        assert state_stack_n >= 1
        assert state_skip >= 1

        super(AugmentEnvWrapper, self).__init__(env)
        self.env = env

        # Amend reward decay if needed
        if amend_reward_decay and state_stack_n != 1:
            import math
            reward_decay = math.pow(reward_decay, 1.0/state_stack_n)

        self.reward_decay, self.reward_offset, self.reward_scale = reward_decay, reward_offset, reward_scale
        self.state_offset, self.state_scale, self.state_stack_n, self.stack_axis, self.state_skip = \
            state_offset, state_scale, state_stack_n, state_stack_axis, state_skip
        self.state_augment_proc, self.reward_shaping_proc = state_augment_proc, reward_shaping_proc
        self.discard_skipped_frames = discard_skipped_frames
        self.random_start = random_start

        # Continues action
        self.is_continuous_action = env.action_space.__class__.__name__ == "Box"
        if self.is_continuous_action:
            if action_limit is None:
                action_limit = [env.action_space.low, env.action_space.high]
            self.action_limit = action_limit
            self.action_scale = (action_limit[1] - action_limit[0])/2.0
            self.action_offset = (action_limit[1] + action_limit[0])/2.0
            logging.warning("limit:%s, scale:%s, offset:%s", action_limit, self.action_scale, self.action_offset)

        # Initialize state stack queue
        if discard_skipped_frames:
            stack_size = state_stack_n
        else:
            stack_size = state_stack_n*state_skip
        self.last_stacked_states = deque(maxlen=stack_size)  # lazy init

        # Amend observation space
        space = env.observation_space
        state_low, state_high = space.low, space.high
        state_shape = list(self.augment_state(self.env.reset()).shape)  # Process the first frame to get observation shape
        state_shape[self.stack_axis] *= stack_size  # Deal with stack axis
        self.observation_space = gym.spaces.box.Box(np.min(state_low), np.max(state_high), state_shape)
        self.state_shape = state_shape

    def __getattr__(self, name):
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

    def do_frame_skip(self, action):
        total_reward = 0.0
        for i in range(self.state_skip):
            observation, reward, done, info = self.env.step(action)
            reward = self.augment_reward(reward, observation=observation, done=done, info=info)
            total_reward += reward

            if not self.discard_skipped_frames:
                self.last_stacked_states.append(self.augment_state(observation))

            if done:
                break

        return self.augment_state(observation), total_reward, done, info

    def _step(self, action):
        action = self.augment_action(action)  # augment action before apply
        observation, reward, done, info = self.do_frame_skip(action)

        # Stack state
        if self.state_stack_n != 1 or self.state_skip != 1:
            if self.discard_skipped_frames:
                self.last_stacked_states.append(observation)
            observation = np.concatenate(self.last_stacked_states, self.stack_axis)

        return observation, reward, done, info

    def _reset(self):
        state = self.env.reset()
        state = self.augment_state(state)

        # No stack
        if self.state_skip == 1:
            return state

        # Use random action to fill the stack queue
        elif self.random_start:
            self.last_stacked_states.append(state)
            for i in range(self.last_stacked_states.maxlen-1):
                action = self.action_space.sample()
                action = self.augment_action(action)
                observation, reward, done, info = self.env.step(action)
                observation = self.augment_state(observation)
                self.last_stacked_states.append(observation)

        # Use the first frame to fill the stack queue
        else:
            for i in range(self.last_stacked_states.maxlen):
                self.last_stacked_states.append(state)

        return np.concatenate(self.last_stacked_states, self.stack_axis)


# TODO: StateHistoryStackEnvWrapper is redundant
class StateHistoryStackEnvWrapper(object):
    def __init__(self, env,
                 stack_n, stack_axis=-1):
        assert stack_n > 0

        self.env = env
        self.stack_n = stack_n
        self.stack_axis = stack_axis

        self.state_history = None

        # Update observation space
        observation_space_shape = list(self.env.observation_space.shape)
        observation_space_shape[stack_axis] *= stack_n

        observation_space_low = self.env.observation_space.low
        observation_space_high = self.env.observation_space.high

        self.observation_space = gym.spaces.box.Box(np.min(observation_space_low),
                                                    np.max(observation_space_high),
                                                    observation_space_shape)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def step(self, action):
        if not self.state_history:
            self.reset()

        state, reward, done, info = self.env.step(action)

        # Add to history
        self.state_history.append(state)
        del self.state_history[0]

        # Stack history
        state = np.concatenate(self.state_history, axis=self.stack_axis)

        # Stack reward

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        self.state_history = [state]*self.stack_n

        return np.concatenate(self.state_history, axis=self.stack_axis)


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


class C2DEnvWrapper(gym.Wrapper):
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
        super(C2DEnvWrapper, self).__init__(env)
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

    def _step(self, *args, **kwargs):
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


class DownsampledMsPacman(gym.ObservationWrapper):
    def __init__(self, env=None, resize=False):
        super(DownsampledMsPacman, self).__init__(env)
        self._resize = resize
        if self._resize:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 80, 3))
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(171, 160, 3))

    def _observation(self, obs):
        img = np.reshape(obs, [210, 160, 3]).astype(np.float32)
        img = img[0:171, :, :]  # crop the bottom part of the picture
        if self._resize:
            img = cv2.resize(img, (80, 80)) # resize to half
        return img.astype(np.uint8)


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


class ScaledRewards(gym.RewardWrapper):

    def __init__(self, env=None, scale=1.0):
        self.reward_scale = scale
        super(ScaledRewards, self).__init__(env)

    def _reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return self.reward_scale * reward


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._array = None  # created the first time this is converted to array

    def __array__(self, dtype=None):
        if self._array is None:
            out = np.concatenate(self._frames, axis=2)
            if dtype is not None:
                out = out.astype(dtype)
            # self._array = out
        else:
            out = self._array

        return out

    def __reduce__(self):
        # discard self._array
        return self.__class__, (self._frames,)

    @property
    def shape(self):
        if len(self._frames) == 0:
            return []
        return [len(self._frames)] + list(self._frames[0].shape)


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


class RemapFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(RemapFrame, self).__init__(env)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0]/2, shp[1]/2, shp[2]))

    def _observation(self, obs):
        return RemapFrame.process(obs)

    @staticmethod
    def process(frame):
        src_size = (96,96)
        dst_size = (48,48)
        center_src = (70,48)
        center_dst = (33,24)
        linear_part_ratio_dst = 0.1
        k_scale = 0.5 #dst_size[0]/src_size[0] typically, set by hand if needed
        d = 1.0/k_scale

        def remap_core(x, size_s, size_d, c_src, c_dst, lr):
            lp      = c_dst - c_dst*lr
            lp_src  = c_src - c_dst*lr
            hp      = c_dst + (size_d - c_dst)*lr
            hp_src  = c_src + (size_d - c_dst)*lr
            a1      = -(lp_src-d*lp) / (lp*lp) # -(lp_src-lp) / (lp*lp)
            b1      = d - 2*a1*lp #add d
            # a2      = (hp_src-hp - size_s + size_d) / (-(hp-size_d)*(hp-size_d))
            a2      = (hp_src-d*hp - size_s + d*size_d) / (-(hp-size_d)*(hp-size_d)) # add d
            b2      = d - 2*a2*hp # add d, 1-2a*hp
            c2      = hp_src - a2*hp*hp-b2*hp
            if x < lp :
                y = a1*x*x + b1*x
            elif x < hp:
                y = x + (c_src - c_dst)
            else:
                y = a2*x*x + b2*x + c2
            return y

        def fx(x):
            return remap_core(x, src_size[0], dst_size[0], center_src[0], center_dst[0], linear_part_ratio_dst)

        def fy(y):
            return remap_core(y, src_size[1], dst_size[1], center_src[1], center_dst[1], linear_part_ratio_dst)

        mapx = np.zeros((dst_size[1], dst_size[0]), dtype=np.float32)
        mapy = np.zeros((dst_size[1], dst_size[0]), dtype=np.float32)

        for x in range(dst_size[0]):
            tmp = fx(x)
            for y in range(dst_size[1]):
                mapx[x][y] = tmp
        for y in range(dst_size[1]):
            tmp = fy(y)
            for x in range(dst_size[0]):
                mapy[x][y] = tmp
        """
        # normalize map to the src image size, d(srctodst) will be affected by ratio
        map_max = mapx.max()
        map_min = mapx.min()
        ratio = (src_size[0]-1)/(map_max - map_min)
        mapx = ratio*(mapx-map_min)
        map_max = mapy.max()
        map_min = mapy.min()
        ratio = (src_size[1]-1)/(map_max - map_min)
        mapy = ratio*(mapy-map_min)
        """
        # remap
        dst = cv2.remap(np.asarray(frame), mapy, mapx, cv2.INTER_LINEAR)
        # for display
        last_frame = dst[:,:,0:3]
        cv2.imshow("image1", cv2.resize(last_frame, (320,320), interpolation=cv2.INTER_LINEAR))
        cv2.waitKey(10)
        return dst


class HalfFrame(gym.ObservationWrapper): #as compare to remapframe
    def __init__(self, env=None):
        super(HalfFrame, self).__init__(env)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0]/2, shp[1]/2, shp[2]))

    def _observation(self, obs):
        return HalfFrame.process(obs)

    @staticmethod
    def process(frame):
        dst = np.asarray(frame)
        #last_frame = np.asarray(frame)[:,:,0:3]
        #cv2.imshow("image0", last_frame)
        #cv2.waitKey(10)
        last_frame = dst[:,:,0:3]
        cv2.imshow("image1", cv2.resize(last_frame, (320,320), interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(10)
        dst=cv2.resize(dst, (48,48), interpolation=cv2.INTER_CUBIC)
        return dst


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
