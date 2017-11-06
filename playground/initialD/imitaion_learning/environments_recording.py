#
# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import gym
import numpy as np
import tensorflow as tf
import cv2
from collections import deque


class EnvRecordingRunner(object):
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
        super(EnvRecordingRunner, self).__init__()
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
        self.savedir = savedir
        self.render_once = True if render_once else False

    def step(self, evaluate=False, eps_dir=None, recording_file=None):
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
        img = self.state[:, :, -3:] * 255.0
        # print "orig: ", img
        img = img.astype(np.uint8)
        # print "astype: ", img
        cv2.imwrite(eps_dir+"/"+str(self.step_n).zfill(6)+"_"
                    +str(self.action)+".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        next_state, reward, done, info = self.env.step(self.action)
        self.total_reward = reward + self.reward_decay * self.total_reward
        recording_file.write("{}, {}, {:.6f}, {:.6f}".format(self.step_n, self.action, reward, self.total_reward))
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
        os.mkdir(self.savedir)
        for i in range(n):
            self.episode_n += 1
            eps_dir = self.savedir+"/"+str(self.episode_n).zfill(6)
            os.mkdir(eps_dir)
            recording_file = open(eps_dir+"/000000.txt", "w")
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
                terminate = self.step(evaluate, eps_dir, recording_file)
                if terminate:
                    break
            recording_file.close()
            logging.warning("Episode %d finished after %d steps, total reward=%f", self.episode_n, t + 1,
                                self.total_reward)
            self.record({"episode_total_reward": self.total_reward})
