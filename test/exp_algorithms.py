#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import logging

import cv2
import gym
import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

import hobotrl as hrl
from hobotrl.experiment import Experiment
from hobotrl.environments import *
import hobotrl.algorithms.ac as ac
import hobotrl.algorithms.dqn as dqn
import hobotrl.algorithms.per as per
import hobotrl.algorithms.dpg as dpg
import playground.optimal_tighten as play_ot
import hobotrl.algorithms.ot as ot
import playground.a3c_onoff as a3coo


class DQNExperiment(Experiment):

    def __init__(self, env,
                 f_create_net,
                 episode_n=1000,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-3),
                 target_sync_rate=0.01,
                 update_interval=1,
                 target_sync_interval=10,
                 max_gradient=10.0,
                 epsilon=0.2,
                 gamma=0.9,
                 greedy_policy=True,
                 ddqn=False,
                 batch_size=8,
                 replay_capacity=1000):
        self.env, self.f_create_net, self.episode_n, \
            self.optimizer_ctor, self.target_sync_rate, self.max_gradient, \
            self.update_interval, self.target_sync_interval, \
            self.epsilon, self.gamma, self.greedy_policy, self.ddqn, \
            self.batch_size, self.replay_capacity = env, f_create_net, episode_n, optimizer_ctor, \
                                                    target_sync_rate, max_gradient, update_interval, \
                                                    target_sync_interval, epsilon, gamma, \
                                                    greedy_policy, ddqn, batch_size, replay_capacity
        super(DQNExperiment, self).__init__()

    def run(self, args):

        state_shape = list(self.env.observation_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(self.env.action_space.n),
            epsilon=self.epsilon,
            # DeepQFuncMixin params
            dqn_param_dict={
                'gamma': self.gamma,
                'f_net': self.f_create_net,
                'state_shape': state_shape,
                'num_actions': self.env.action_space.n,
                'training_params': (self.optimizer_ctor(), self.target_sync_rate, self.max_gradient),
                'schedule': (self.update_interval, self.target_sync_interval),
                'greedy_policy': self.greedy_policy,
                'ddqn': self.ddqn,
            },
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
            buffer_param_dict={
                "capacity": self.replay_capacity,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (),
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                 }},
            batch_size=self.batch_size,
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(
            graph=tf.get_default_graph(), worker_index=0,
            init_op=tf.global_variables_initializer(), save_dir=args.logdir
        )
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(
                self.env, agent, evaluate_interval=sys.maxint,
                render_interval=sys.maxint, logdir=args.logdir
            )
            runner.episode(self.episode_n)


class PERDQNExperiment(Experiment):

    def __init__(self, env,
                 f_create_net,
                 episode_n=1000,
                 priority_bias=0.5,  # todo search what combination of exponent/importance_correction works better
                 importance_weight=hrl.utils.CappedLinear(1e6, 0.5, 1.0),
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-3),
                 target_sync_rate=0.01,
                 update_interval=1,
                 target_sync_interval=10,
                 max_gradient=10.0,
                 epsilon=0.2,
                 gamma=0.9,
                 greedy_policy=True,
                 ddqn=False,
                 batch_size=8,
                 replay_capacity=1000):
        self.env, self.f_create_net, self.episode_n, \
            self.optimizer_ctor, self.target_sync_rate, self.max_gradient, \
            self.update_interval, self.target_sync_interval, \
            self.epsilon, self.gamma, self.greedy_policy, self.ddqn, \
            self.batch_size, self.replay_capacity = env, f_create_net, episode_n, optimizer_ctor, \
                                                    target_sync_rate, max_gradient, update_interval, \
                                                    target_sync_interval, epsilon, gamma, \
                                                    greedy_policy, ddqn, batch_size, replay_capacity
        self.priority_bias, self.importance_weight = priority_bias, importance_weight

        super(PERDQNExperiment, self).__init__()

    def run(self, args):
        state_shape = list(self.env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = per.PrioritizedDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(self.env.action_space.n),
            epsilon=self.epsilon,
            # DeepQFuncMixin params
            dqn_param_dict={
                'gamma': 0.9,
                'f_net': self.f_create_net,
                'state_shape': state_shape,
                'num_actions': self.env.action_space.n,
                'training_params': (self.optimizer_ctor(), self.target_sync_rate, self.max_gradient),
                'schedule': (self.update_interval, self.target_sync_interval),
                'greedy_policy': self.greedy_policy,
                'ddqn': self.ddqn,
            },

            # ReplayMixin params
            buffer_class=hrl.playback.NearPrioritizedPlayback,
            buffer_param_dict={
                "capacity": self.replay_capacity,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (),
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                },
                "priority_bias": self.priority_bias,
                "importance_weight": self.importance_weight,
            },
            batch_size=self.batch_size,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self.env, agent, evaluate_interval=sys.maxint, render_interval=sys.maxint,
                                        logdir=args.logdir)
            runner.episode(self.episode_n)


class DPGExperiment(Experiment):

    def __init__(self, env,
                 f_net_ddp,
                 f_net_dqn,
                 episode_n=1000,
                 optimizer_ddp_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-3),
                 optimizer_dqn_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-3),
                 target_sync_rate=0.001,
                 ddp_update_interval=1,
                 ddp_sync_interval=1,
                 dqn_update_interval=1,
                 dqn_sync_interval=1,
                 max_gradient=10.0,
                 ou_params=(0.0, 0.15, 0.2),
                 gamma=0.99,
                 batch_size=8,
                 replay_capacity=1000):
        self.env, self.f_net_ddp, self.f_net_dqn, self.episode_n, \
            self.optimizer_ddp_ctor, self.optimizer_dqn_ctor, self.target_sync_rate, \
            self.ddp_update_interval, self.ddp_sync_interval, \
            self.dqn_update_interval, self.dqn_sync_interval, \
            self.max_gradient, \
            self.gamma, self.ou_params, \
            self.batch_size, self.replay_capacity = env, f_net_ddp, f_net_dqn, episode_n, \
                                                    optimizer_ddp_ctor, optimizer_dqn_ctor, target_sync_rate, \
                                                    ddp_update_interval, ddp_sync_interval, \
                                                    dqn_update_interval, dqn_sync_interval, \
                                                    max_gradient, \
                                                    gamma, ou_params, \
                                                    batch_size, replay_capacity
        super(DPGExperiment, self).__init__()

    def run(self, args):
        training_params_ddp = (self.optimizer_ddp_ctor(), self.target_sync_rate, self.max_gradient)
        training_params_dqn = (self.optimizer_dqn_ctor(), self.target_sync_rate, self.max_gradient)
        schedule_ddp = (self.ddp_update_interval, self.ddp_sync_interval)
        schedule_dqn = (self.dqn_update_interval, self.dqn_sync_interval)
        state_shape = list(self.env.observation_space.shape)
        action_shape = list(self.env.action_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = dpg.DPG(
            # === ReplayMixin params ===
            buffer_class=hrl.playback.MapPlayback,
            buffer_param_dict={
                "capacity": self.replay_capacity,
                "sample_shapes": {
                    'state': state_shape,
                    'action': action_shape,
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                 }},
            batch_size=self.batch_size,
            # === OUExplorationMixin ===
            ou_params=self.ou_params,
            action_shape=action_shape,
            # === DeepDeterministicPolicyMixin ===
            ddp_param_dict={
                'f_net': self.f_net_ddp,
                'state_shape': state_shape,
                'action_shape': action_shape,
                'training_params': training_params_ddp,
                'schedule': schedule_ddp,
                'graph': tf.get_default_graph()
            },
            # === DeepQFuncMixin params ===
            dqn_param_dict={
                'gamma': self.gamma,
                'f_net': self.f_net_dqn,
                'state_shape': state_shape,
                'action_shape': action_shape,
                'training_params': training_params_dqn,
                'schedule': schedule_dqn,
                'greedy_policy': True,
                'graph': tf.get_default_graph()
            },
            is_action_in=True
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(
            graph=tf.get_default_graph(), worker_index=0,
            init_op=tf.global_variables_initializer(), save_dir=args.logdir
        )
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(
                self.env, agent, evaluate_interval=sys.maxint, render_interval=40, 
                render_once=True,
                logdir=args.logdir
            )
            runner.episode(self.episode_n)


class ACOOExperiment(Experiment):
    def __init__(self, env,
                 f_create_net,
                 episode_n=10000,
                 reward_decay=0.99,
                 on_batch_size=32,
                 off_batch_size=32,
                 off_interval=8,
                 sync_interval=1000,
                 replay_size=10000,
                 prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(4e5, 1e-2, 1e-3),
                 l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(1e-4),
                 ddqn=False,
                 aux_r=False,
                 aux_d=False,
                 ):
        super(ACOOExperiment, self).__init__()
        self.env = env
        self.f_create_net = f_create_net
        self.episode_n = episode_n
        self.reward_decay = reward_decay
        self.on_batch_size = on_batch_size
        self.off_batch_size = off_batch_size
        self.off_interval = off_interval
        self.sync_interval = sync_interval
        self.replay_size = replay_size
        self.prob_min = prob_min
        self.entropy = entropy
        self.l2 = l2
        self.optimizer_ctor = optimizer_ctor
        self.ddqn = ddqn
        self.aux_r = aux_r
        self.aux_d = aux_d

    def run(self, args):
        logging.warning("after run")
        cluster = eval(args.cluster)
        cluster_spec = tf.train.ClusterSpec(cluster)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        server = tf.train.Server(cluster_spec,
                                 job_name=args.job,
                                 task_index=args.index,
                                 config=config)
        worker_n = len(cluster["worker"])
        if args.job == "ps":
            logging.warning("starting ps server")
            server.join()
        else:
            kwargs = {"ddqn": self.ddqn, "aux_r": self.aux_r, "aux_d": self.aux_d, "reward_decay": self.reward_decay}
            env = self.env
            state_shape = list(self.env.observation_space.shape)
            with tf.device("/job:worker/task:0"):
                global_step = tf.get_variable('global_step', [],
                                              initializer=tf.constant_initializer(0),
                                              trainable=False)
                global_net = a3coo.ActorCritic(0, "global_net", state_shape, env.action_space.n,
                                               self.f_create_net, optimizer=self.optimizer_ctor(),
                                               global_step=global_step, **kwargs)

            for i in range(worker_n):
                with tf.device("/job:worker/task:%d" % i):
                    worker = a3coo.A3CAgent(
                        index=i,
                        parent_net=global_net,
                        create_net=self.f_create_net,
                        state_shape=state_shape,
                        num_actions=env.action_space.n,
                        replay_capacity=self.replay_size,
                        train_on_interval=self.on_batch_size,
                        train_off_interval=self.off_interval,
                        target_follow_interval=self.sync_interval,
                        off_batch_size=self.off_batch_size,
                        entropy=self.entropy,
                        global_step=global_step,
                        optimizer=self.optimizer_ctor(),
                        **kwargs
                    )
                    if i == args.index:
                        agent = worker

            sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=args.index,
                                       init_op=tf.global_variables_initializer(), save_dir=args.logdir)
            with sv.prepare_or_wait_for_session(server.target) as sess:
                agent.set_session(sess)
                runner = hrl.envs.EnvRunner(env, agent, reward_decay=self.reward_decay,
                                            evaluate_interval=sys.maxint, render_interval=sys.maxint,
                                            render_once=True,
                                            logdir=args.logdir if args.index == 0 else None)
                runner.episode(self.episode_n)


