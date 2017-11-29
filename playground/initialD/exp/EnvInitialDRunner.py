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


def func_compile_action(action):
    ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
    return ALL_ACTIONS[action]



def func_compile_exp_agent(action, rewards, done):
    # Compile reward
    rewards = np.mean(np.array(rewards), axis=0)
    rewards = rewards.tolist()
    rewards.append(np.logical_or(action==1, action==2))  # action == turn?
    print (' '*5+'R: ['+'{:4.2f} '*len(rewards)+']').format(*rewards),

    road_change = rewards[1] > 0.01  # road changed
    road_invalid = rewards[0] > 0.01  # any yellow or red
    if road_change and not last_road_change:
        road_invalid_at_enter = road_invalid
    last_road_change = road_change
    speed = rewards[2]
    obs_risk = rewards[5]

    ema_speed = 0.5*ema_speed + 0.5*speed
    ema_dist = 1.0 if rewards[6] > 2.0 else 0.9 * ema_dist
    momentum_opp = (rewards[3]<0.5)*(momentum_opp+(1-rewards[3]))
    momentum_opp = min(momentum_opp, 20)
    momentum_ped = (rewards[4]>0.5)*(momentum_ped+rewards[4])
    momentum_ped = min(momentum_ped, 12)

    # road_change
    rewards[0] = -100*(
        (road_change and ema_dist>0.2) or (road_change and momentum_ped > 0)
    )*(n_skip-cnt_skip)  # direct penalty
    # velocity
    rewards[2] *= 10
    rewards[2] -= 10
    # opposite
    rewards[3] = -20*(0.9+0.1*momentum_opp)*(momentum_opp>1.0)
    # ped
    rewards[4] = -40*(0.9+0.1*momentum_ped)*(momentum_ped>1.0)
    # obs factor
    rewards[5] *= -100.0
    # dist
    rewards[6] *= 0.0
    # steering
    rewards[-1] *= -40  # -3
    reward = np.sum(rewards)/100.0
    print '{:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}'.format(
        road_invalid_at_enter, momentum_opp, momentum_ped, ema_dist),
    print ': {:5.2f}'.format(reward)

    if road_invalid_at_enter:
        print "[Early stopping] entered invalid road."
        road_invalid_at_enter = False
        # done = True

    if road_change and ema_dist > 0.2:
        print "[Early stopping] turned onto intersection."
        done = True

    if road_change and momentum_ped>0:
        print "[Early stopping] ped onto intersection."
        done = True

    if obs_risk > 1.0:
        print "[Early stopping] hit obstacle."
        done = True

    return reward, done


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
        self.episode_n, self.step_n = 0, 0
        self.state = None
        self.action = None
        self.total_reward = 0.0
        self.summary_writer = None
        if logdir is not None:
            self.summary_writer = SummaryWriterCache.get(logdir)
        self.steps_saver = None
        if savedir is not None:
            self.steps_saver = StepsSaver(savedir)
        self.flag_success = None

    def step(self):
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
        self.flag_success = True if done and reward > 0.0 else False
        if self.savedir is not None:
            self.steps_saver.save(self.episode_n, self.step_n, self.state, self.action,
                                vec_reward, reward, done, self.total_reward, self.flag_success)
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
        self.steps_saver.end_save()
        return rewards
