# -*- coding: utf-8 -*-

import sys
sys.path.insert('./')
sys.path.index('../../..')
import os
import time
import logging
import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
import cv2
from collections import deque
from playground.initialD.ros_environments.clients import DrivingSimulatorEnvClient as DrivingSimulatorEnv


def func_compile_obs(obss):
    obs1 = obss[-1][0]
    obs2 = obss[-2][0]
    obs3 = obss[-3][0]
    print obss[-1][1]
    # cast as uint8 is important otherwise float64
    obs = ((obs1 + obs2)/2).astype('uint8')
    # print obs.shape
    return obs.copy()

ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
AGENT_ACTIONS = ALL_ACTIONS[:3]


def func_compile_exp_agent(action, rewards, done):
    pass


class StepsSaver(object):
    def __init__(self, savedir):
        self.savedir = savedir
        self.eps_dir = None
        self.file = None
        self.total_file = open(self.savedir + "/0000.txt", "w")

    def end_save(self):
        self.stat_file.close()

    def parse_state(self):
        return np.array(self.state)[:, :, 6:]

    def save(self, n_ep, n_step, state, action, vec_reward, reward,
                  done, cum_reward, flag_success):
        if self.file is None:
            self.eps_dir = self.savedir + "/" + str(n_ep).zfill(4)
            os.mkdir(self.eps_dir)
            self.file = open(self.eps_dir + "/0000.txt", "w")

        img_path = self.eps_dir + "/" + str(n_step + 1).zfill(4) + "_" + str(action) + ".jpg"
        cv2.imwrite(img_path, cv2.cvtColor(self.parse_state(), cv2.COLOR_RGB2BGR))
        self.file.write(str(n_step) + ',' + str(action) + ',' + str(reward) + '\n')
        vec_reward = np.mean(np.array(vec_reward), axis=0)
        vec_reward = vec_reward.tolist()
        str_reward = ""
        for r in vec_reward:
            str_reward += str(r)
            str_reward += ","
        str_reward += "\n"
        self.file.write(str_reward)
        self.file.write("\n")
        if done:
            self.file.close()
            self.file = None
            self.stat_file.write("{}, {}, {}, {}\n".format(n_ep, cum_reward, flag_success, done))


class EnvInitialDRunner(object):
    """
    interaction between agent and environment.
    """
    def __init__(self, env, agent, reward_decay=0.99, max_episode_len=5000,
                 evaluate_interval=sys.maxint, render_interval=sys.maxint,
                 render_once=False,
                 logdir=None, savedir=None):
        """

        :param env: environment.
        :param agent: agent.
        :param reward_decay: deprecated. EnvRunner should not discount future rewards.
        :param max_episode_len:
        :param evaluate_interval:
        :param render_interval:
        :param logdir: dir to save info from agent as tensorboard log.
        """
        super(EnvInitialDRunner, self).__init__()
        self.env, self.agent = env, agent
        self.reward_decay, self.max_episode_len = 1.0, max_episode_len
        self.evaluate_interval, self.render_interval = evaluate_interval, render_interval
        self.episode_n, self.step_n, self.step_t = 0, 0, 0
        self.state = None
        self.action = None
        self.total_reward = 0.0
        self.summary_writer = None
        if logdir is not None:
            self.summary_writer = SummaryWriterCache.get(logdir)
        self.steps_saver = None
        if savedir is not None:
            self.savedir = savedir
            self.steps_saver = StepsSaver(savedir)

    def step(self):
        """
        agent runs one step against env.
        :param evaluate:
        :return:
        """
        self.step_n += 1
        self.step_t += 1
        # TODO: directly calling agent.act will by-pass BaseDeepAgent, which
        # checks and assigns 'sess' arugment. So we manually set sess here. But
        # is there a better way to do this?
        self.action = self.agent.act(
            state=self.state, sess=self.agent.sess
        )
        next_state, vec_reward, done, _ = self.env.step(self.action)
        reward, done = func_compile_exp_agent(self.action, vec_reward, done)
        self.total_reward = reward + self.reward_decay * self.total_reward
        info = self.agent.step(
            state=self.state, action=self.action, reward=reward,
            next_state=next_state, episode_done=done
        )
        self.record(info)
        flag_success = True if done and reward > 0.0 else False
        if self.savedir is not None:
            self.steps_saver.save(self.episode_n, self.step_t, self.state, self.action,
                                vec_reward, reward, done, self.total_reward, flag_success)
        self.state = next_state
        if done:
            self.step_t = 0
        return done

    def record(self, info):
        if self.summary_writer is not None:
            for name in info:
                value = info[name]
                summary = tf.Summary()
                summary.value.add(tag=name, simple_value=np.mean(value))
                self.summary_writer.add_summary(summary, self.total_step)

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
            t = 0
            for t in range(self.max_episode_len):
                terminate = self.step()
                if terminate:
                    break
            logging.warning("Episode %d finished after %d steps, total reward=%f", self.episode_n, t+1,
                                self.total_reward)
            summary = tf.Summary()
            summary.value.add(tag="episode_total_reward", simple_value=self.total_reward)
            summary.value.add(tag="epsisode_success", simple_value=self.flag_success)
            self.summary_writer.add_summary(summary, self.episode_n)
            rewards.append(self.total_reward)
        if self.savedir is not None:
            self.steps_saver.end_save()
        return rewards



def gen_backend_cmds():
    ws_path = '/Projects/catkin_ws/'
    initialD_path = '/Projects/hobotrl/playground/initialD/'
    backend_path = initialD_path + 'ros_environments/backend_scripts/'
    utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
    backend_cmds = [
        # Parse maps
        ['python', utils_path+'parse_map.py',
         ws_path+'src/Map/src/map_api/data/honda_wider.xodr',
         utils_path+'road_segment_info.txt'],
        # Generate obs and launch file
        ['python', utils_path+'gen_launch_dynamic_v1.py',
         utils_path+'road_segment_info.txt', ws_path,
         utils_path+'honda_dynamic_obs_template_tilt.launch',
         32, '--random_n_obs'],
        # start roscore
        ['roscore'],
        # start reward function script
        ['python', backend_path+'gazebo_rl_reward.py'],
        # start road validity node script
        ['python', backend_path+'road_validity.py',
         utils_path+'road_segment_info.txt.signal'],
        # start car_go script
        ['python', backend_path+'car_go.py'],
        # start simulation restarter backend
        ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
        # ['python', backend_path + 'non_stop_data_capture.py', 0]
    ]
    return backend_cmds


def func_compile_action(action):
    ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
    return ALL_ACTIONS[action]


def func_compile_reward(rewards):
    return rewards


class InitialD(gym.Env):
    def __init__(self, host, port):
        self.driving_env = DrivingSimulatorEnv(
            address=host, port=port,
            backend_cmds=gen_backend_cmds(),
            defs_obs=[
                ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
                ('/decision_result', 'std_msgs.msg.Int16')
            ],
            defs_reward=[
                ('/rl/current_road_validity', 'std_msgs.msg.Int16'),
                ('/rl/entering_intersection', 'std_msgs.msg.Bool'),
                ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
                ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
                ('/rl/on_biking_lane', 'std_msgs.msg.Bool'),
                ('/rl/obs_factor', 'std_msgs.msg.Float32'),
                ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
            ],
            defs_action=[('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')],
            rate_action=10.0,
            window_sizes={'obs': 3, 'reward': 3},
            buffer_sizes={'obs': 3, 'reward': 3},
            # should be wrapped
            func_compile_obs=func_compile_obs,
            func_compile_reward=func_compile_reward,
            func_compile_action=func_compile_action,
            step_delay_target=0.5)

    def _reset(self):
        return self.driving_env.reset()

    def _step(self, action):
        return self.driving_env.step(action)


# class ScalarReward(gym.Wrapper):
#     def __init__(self, env):
#         gym.Wrapper.__init__(self, env)
#
#     def _func_reward(self, action, vec_rewards):
#         pass
#
#     def _step(self, action):
#         ob, vec_rewards, done, info = self.env.step(action)
#         reward = self._func_reward(action, vec_rewards)
#         return ob, reward, done, info
