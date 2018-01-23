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
from hobotrl.async import AsynchronousAgent
import hobotrl.sampling as sampling
import hobotrl.algorithms.ac as ac
import hobotrl.algorithms.dqn as dqn
import hobotrl.algorithms.dpg as dpg
import hobotrl.algorithms.ot as ot
import playground.a3c_onoff as a3coo
import playground.a3c_continuous_onoff as a3ccoo
import playground.ot_model as ot_model


class DQNExperiment(Experiment):

    def __init__(self, env,
                 f_create_q,
                 episode_n=1000,
                 discount_factor=0.99,
                 ddqn=False,
                 # target network sync arguments
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 # sampler arguments
                 update_interval=4,
                 replay_size=1000,
                 batch_size=32,
                 # epsilon greedy arguments
                 greedy_epsilon=0.3,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0)
                 ):
        self._env, self._f_create_q, self._episode_n, \
            self._discount_factor, \
            self._ddqn, \
            self._target_sync_interval, \
            self._target_sync_rate, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._greedy_epsilon, \
            self._network_optimizer_ctor = \
            env, f_create_q, episode_n, \
            discount_factor, \
            ddqn, \
            target_sync_interval, \
            target_sync_rate, \
            update_interval, \
            replay_size, \
            batch_size, \
            greedy_epsilon, \
            network_optimizer_ctor

        super(DQNExperiment, self).__init__()

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = dqn.DQN(
            f_create_q=self._f_create_q,
            state_shape=state_shape,
            # OneStepTD arguments
            num_actions=self._env.action_space.n,
            discount_factor=self._discount_factor,
            ddqn=self._ddqn,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            update_interval=self._update_interval,
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            # epsilon greedy arguments
            greedy_epsilon=self._greedy_epsilon,
            network_optimizer=self._network_optimizer_ctor(),
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=True,
            )
            runner.episode(self._episode_n)


class ADQNExperiment(Experiment):

    def __init__(self, env,
                 f_create_q,
                 episode_n=1000,
                 discount_factor=0.99,
                 ddqn=False,
                 # target network sync arguments
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 # sampler arguments
                 update_interval=4,
                 replay_size=1000,
                 batch_size=32,
                 # epsilon greedy arguments
                 greedy_epsilon=0.3,
                 learning_rate=1e-4,
                 ):
        self._env, self._f_create_q, self._episode_n, \
            self._discount_factor, \
            self._ddqn, \
            self._target_sync_interval, \
            self._target_sync_rate, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._greedy_epsilon, \
            self._learning_rate = \
            env, f_create_q, episode_n, \
            discount_factor, \
            ddqn, \
            target_sync_interval, \
            target_sync_rate, \
            update_interval, \
            replay_size, \
            batch_size, \
            greedy_epsilon, \
            learning_rate

        super(ADQNExperiment, self).__init__()

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)

        def create_optimizer():
            return tf.train.AdamOptimizer(self._learning_rate)

        def create_agent(n_optimizer, global_step):
            # all ScheduledParam hyper parameters are mutable objects.
            # so we will not want to use same object for different Agent instances.
            self._greedy_epsilon = hrl.utils.clone_params(self._greedy_epsilon)
            agent = dqn.DQN(
                f_create_q=self._f_create_q,
                state_shape=state_shape,
                # OneStepTD arguments
                num_actions=self._env.action_space.n,
                discount_factor=self._discount_factor,
                ddqn=self._ddqn,
                # target network sync arguments
                target_sync_interval=self._target_sync_interval,
                target_sync_rate=self._target_sync_rate,
                # sampler arguments
                update_interval=self._update_interval,
                replay_size=self._replay_size,
                batch_size=self._batch_size,
                # epsilon greedy arguments
                greedy_epsilon=self._greedy_epsilon,
                network_optimizer=n_optimizer,
                global_step=global_step
            )
            return agent

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.create_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor,  max_episode_len=1000,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=args.render_once,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(self._episode_n)


class PERDQNExperiment(Experiment):

    def __init__(self, env,
                 f_create_q,
                 episode_n=1000,
                 discount_factor=0.99,
                 ddqn=False,
                 # target network sync arguments
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 # sampler arguments
                 update_interval=4,
                 replay_size=1000,
                 batch_size=32,
                 priority_bias=0.5,
                 importance_weight=0.5,
                 # epsilon greedy arguments
                 greedy_epsilon=0.3,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0)
                 ):
        self._env, self._f_create_q, self._episode_n, \
            self._discount_factor, \
            self._ddqn, \
            self._target_sync_interval, \
            self._target_sync_rate, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._greedy_epsilon, \
            self._network_optimizer_ctor = \
            env, f_create_q, episode_n, \
            discount_factor, \
            ddqn, \
            target_sync_interval, \
            target_sync_rate, \
            update_interval, \
            replay_size, \
            batch_size, \
            greedy_epsilon, \
            network_optimizer_ctor
        self._priority_bias, self._importance_weight = priority_bias, importance_weight
        self._sampler = sampling.TransitionSampler(
            hrl.playback.NearPrioritizedPlayback(replay_size, priority_bias, importance_weight))

        super(PERDQNExperiment, self).__init__()

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        kwargs = {"priority_bias": self._priority_bias, "importance_weight": self._importance_weight}
        agent = dqn.DQN(
            f_create_q=self._f_create_q,
            state_shape=state_shape,
            # OneStepTD arguments
            num_actions=self._env.action_space.n,
            discount_factor=self._discount_factor,
            ddqn=self._ddqn,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            sampler=self._sampler,
            update_interval=self._update_interval,
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            # epsilon greedy arguments
            greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.5, 0.1),
            network_optimizer=self._network_optimizer_ctor(),
            global_step=global_step,
            **kwargs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(self._env, agent,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        logdir=args.logdir)
            runner.episode(self._episode_n)


class OTDQNExperiment(Experiment):

    def __init__(self, env,
                 f_create_q,
                 episode_n=1000,
                 discount_factor=0.99,
                 ddqn=False,
                 # target network sync arguments
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 # sampler arguments
                 update_interval=4,
                 replay_size=1000,
                 batch_size=8,
                 lower_weight=1.0,
                 upper_weight=1.0,
                 neighbour_size=8,
                 # epsilon greedy arguments
                 greedy_epsilon=0.3,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 **kwargs
                 ):
        self._env, self._f_create_q, self._episode_n, \
            self._discount_factor, \
            self._ddqn, \
            self._target_sync_interval, \
            self._target_sync_rate, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._greedy_epsilon, \
            self._network_optimizer_ctor, \
            self._lower_weight, self._upper_weight, self._neighbour_size = \
            env, f_create_q, episode_n, \
            discount_factor, \
            ddqn, \
            target_sync_interval, \
            target_sync_rate, \
            update_interval, \
            replay_size, \
            batch_size, \
            greedy_epsilon, \
            network_optimizer_ctor, \
            lower_weight, upper_weight, neighbour_size
        self._kwargs = kwargs

        super(OTDQNExperiment, self).__init__()

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = hrl.algorithms.ot.OTDQN(
            f_create_q=self._f_create_q,
            lower_weight=self._lower_weight,
            upper_weight=self._upper_weight,
            neighbour_size=self._neighbour_size,
            state_shape=state_shape,
            # OneStepTD arguments
            num_actions=self._env.action_space.n,
            discount_factor=self._discount_factor,
            ddqn=self._ddqn,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            update_interval=self._update_interval,
            replay_size=self._replay_size,
            batch_size=self._batch_size,
            # epsilon greedy arguments
            greedy_epsilon=self._greedy_epsilon,
            network_optmizer=self._network_optimizer_ctor(),
            global_step=global_step,
            **self._kwargs
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=sys.maxint, logdir=args.logdir,
                render_once=True,
            )
            return runner.episode(self._episode_n)


class AOTDQNExperiment(Experiment):
    def __init__(self,
                 env,
                 f_create_q,
                 episode_n=1000,
                 discount_factor=0.99,
                 ddqn=False,
                 # target network sync arguments
                 target_sync_interval=100,
                 target_sync_rate=1.0,
                 # sampler arguments
                 update_interval=4,
                 replay_size=1000,
                 batch_size=8,
                 lower_weight=1.0,
                 upper_weight=1.0,
                 neighbour_size=8,
                 # epsilon greedy arguments
                 greedy_epsilon=0.3,
                 learning_rate=1e-4,
                 sampler_creator=None,
                 **kwargs
                 ):
        super(AOTDQNExperiment, self).__init__()
        self._env, self._f_create_q, self._episode_n, \
            self._discount_factor, \
            self._ddqn, \
            self._target_sync_interval, \
            self._target_sync_rate, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._greedy_epsilon, \
            self._learning_rate, \
            self._lower_weight, self._upper_weight, self._neighbour_size = \
            env, f_create_q, episode_n, \
            discount_factor, \
            ddqn, \
            target_sync_interval, \
            target_sync_rate, \
            update_interval, \
            replay_size, \
            batch_size, \
            greedy_epsilon, \
            learning_rate, \
            lower_weight, upper_weight, neighbour_size
        if sampler_creator is None:
            def f(args):
                max_traj_length = 200
                sampler = sampling.TruncateTrajectorySampler2(None, replay_size / max_traj_length, max_traj_length,
                                                              batch_size, neighbour_size, update_interval)
                return sampler
            sampler_creator = f
        self._sampler_creator = sampler_creator
        self._kwargs = kwargs

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)

        def create_optimizer():
            return tf.train.AdamOptimizer(self._learning_rate)

        def create_agent(n_optimizer, global_step):
            # all ScheduledParam hyper parameters are mutable objects.
            # so we will not want to use same object for different Agent instances.
            self._greedy_epsilon = hrl.utils.clone_params(self._greedy_epsilon)
            sampler = self._sampler_creator(args)
            agent = hrl.algorithms.ot.OTDQN(
                f_create_q=self._f_create_q,
                lower_weight=self._lower_weight,
                upper_weight=self._upper_weight,
                neighbour_size=self._neighbour_size,
                state_shape=state_shape,
                # OneStepTD arguments
                num_actions=self._env.action_space.n,
                discount_factor=self._discount_factor,
                ddqn=self._ddqn,
                # target network sync arguments
                target_sync_interval=self._target_sync_interval,
                target_sync_rate=self._target_sync_rate,
                # sampler arguments
                update_interval=self._update_interval,
                replay_size=self._replay_size,
                batch_size=self._batch_size,
                # epsilon greedy arguments
                greedy_epsilon=self._greedy_epsilon,
                network_optmizer=n_optimizer,
                global_step=global_step,
                sampler=sampler,
                **self._kwargs
            )
            return agent

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.create_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor,  max_episode_len=1000,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=args.render_once,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(self._episode_n)


class DPGExperiment(Experiment):

    def __init__(self, env,
                 f_se, f_actor, f_critic,
                 episode_n=1000,
                 # ACUpdate arguments
                 discount_factor=0.9,
                 # optimizer arguments
                 network_optimizer_ctor=lambda:hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 # policy arguments
                 ou_params=(0, 0.2, 0.2),
                 # target network sync arguments
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 # sampler arguments
                 batch_size=32,
                 replay_capacity=1000):
        self._env, self._f_se, self._f_actor, self._f_critic, self._episode_n,\
            self._discount_factor, self._network_optimizer_ctor, \
            self._ou_params, self._target_sync_interval, self._target_sync_rate, \
            self._batch_size, self._replay_capacity = \
            env, f_se, f_actor, f_critic, episode_n, \
            discount_factor, network_optimizer_ctor, \
            ou_params, target_sync_interval, target_sync_rate, \
            batch_size, replay_capacity
        super(DPGExperiment, self).__init__()

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)
        dim_action = self._env.action_space.shape[-1]
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = hrl.DPG(
            f_se=self._f_se,
            f_actor=self._f_actor,
            f_critic=self._f_critic,
            state_shape=state_shape,
            dim_action=dim_action,
            # ACUpdate arguments
            discount_factor=self._discount_factor,
            target_estimator=None,
            # optimizer arguments
            network_optimizer=self._network_optimizer_ctor(),
            # policy arguments
            ou_params=self._ou_params,
            # target network sync arguments
            target_sync_interval=self._target_sync_interval,
            target_sync_rate=self._target_sync_rate,
            # sampler arguments
            sampler=hrl.sampling.TransitionSampler(hrl.playback.MapPlayback(self._replay_capacity), self._batch_size),
            batch_size=self._batch_size,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint, render_interval=40,
                render_once=True,
                logdir=args.logdir
            )
            runner.episode(self._episode_n)


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

            with agent.create_session(master=server.target, worker_index=args.index, save_dir=args.logdir) as sess:
                runner = hrl.envs.EnvRunner(env, agent, reward_decay=self.reward_decay,
                                            evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                            render_once=True,
                                            logdir=args.logdir if args.index == 0 else None)
                runner.episode(self.episode_n)


class ACOOExperimentCon(Experiment):
    def __init__(self, env,
                 f_create_net,
                 episode_n=10000,
                 reward_decay=0.99,
                 entropy_scale=1,
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
        super(ACOOExperimentCon, self).__init__()
        self.env = env
        self.f_create_net = f_create_net
        self.episode_n = episode_n
        self.reward_decay = reward_decay
        self.entropy_scale = entropy_scale
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
            kwargs = {"ddqn": self.ddqn, "aux_r": self.aux_r, "aux_d": self.aux_d, "reward_decay": self.reward_decay,
                      "entropy_scale": self.entropy_scale}
            env = self.env
            state_shape = list(self.env.observation_space.shape)
            with tf.device("/job:worker/task:0"):
                global_step = tf.get_variable('global_step', [],
                                              initializer=tf.constant_initializer(0),
                                              trainable=False)
                global_net = a3ccoo.ActorCritic(0, "global_net", state_shape, env.action_space.shape[0],
                                               self.f_create_net, optimizer=self.optimizer_ctor(),
                                               global_step=global_step, **kwargs)

            for i in range(worker_n):
                with tf.device("/job:worker/task:%d" % i):
                    worker = a3ccoo.A3CAgent(
                        index=i,
                        parent_net=global_net,
                        create_net=self.f_create_net,
                        state_shape=state_shape,
                        num_actions=env.action_space.shape[0],
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

            with agent.create_session(master=server.target, worker_index=args.index, save_dir=args.logdir) as sess:
                agent.set_session(sess)
                runner = hrl.envs.EnvRunner(env, agent, reward_decay=self.reward_decay,
                                            evaluate_interval=sys.maxint, render_interval=sys.maxint,
                                            render_once=True,
                                            logdir=args.logdir if args.index == 0 else None)
                runner.episode(self.episode_n)


class ACExperiment(Experiment):

    def __init__(self,
                 env, f_create_net, episode_n=1000,
                 discount_factor=0.9,
                 entropy=1e-2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 batch_size=8
                 ):
        super(ACExperiment, self).__init__()
        self._env, self._f_create_net, self._episode_n, \
            self._discount_factor, self._entropy, self._network_optimizer_ctor, self._batch_size = \
            env, f_create_net, episode_n, \
            discount_factor, entropy, network_optimizer_ctor, batch_size

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)

        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
        agent = hrl.ActorCritic(
            f_create_net=self._f_create_net,
            state_shape=state_shape,
            # ACUpdate arguments
            discount_factor=self._discount_factor,
            entropy=self._entropy,
            target_estimator=None,
            max_advantage=100.0,
            # optimizer arguments
            network_optimizer=self._network_optimizer_ctor(),
            max_gradient=10.0,
            # sampler arguments
            sampler=None,
            batch_size=self._batch_size,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir
            )
            runner.episode(self._episode_n)


class A3CExperiment(Experiment):
    def __init__(self,
                 env, f_create_net, episode_n=1000,
                 learning_rate=1e-4,
                 discount_factor=0.9,
                 entropy=1e-2,
                 batch_size=8
                 ):
        super(A3CExperiment, self).__init__()
        self._env, self._f_create_net, self._episode_n, self._learning_rate, \
            self._discount_factor, self._entropy, self._batch_size = \
            env, f_create_net, episode_n, learning_rate, \
            discount_factor, entropy, batch_size

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)

        def create_optimizer():
            return tf.train.AdamOptimizer(self._learning_rate)

        def create_agent(n_optimizer, global_step):
            # all ScheduledParam hyper parameters are mutable objects.
            # so we will not want to use same object for different Agent instances.
            entropy = hrl.utils.clone_params(self._entropy)
            agent = hrl.ActorCritic(
                f_create_net=self._f_create_net,
                state_shape=state_shape,
                # ACUpdate arguments
                discount_factor=self._discount_factor,
                entropy=entropy,
                target_estimator=None,
                max_advantage=100.0,
                # optimizer arguments
                network_optimizer=n_optimizer,
                # sampler arguments
                sampler=None,
                batch_size=self._batch_size,

                global_step=global_step,
            )
            return agent

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.create_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor,  max_episode_len=1000,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=args.render_once,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(self._episode_n)


class A3CExperimentWithICM(Experiment):
    def __init__(self,
                 env, f_se, f_ac, f_forward, f_inverse,
                 episode_n=1000,
                 learning_rate=1e-4,
                 discount_factor=0.9,
                 entropy=1e-2,
                 batch_size=8
                 ):
        super(A3CExperimentWithICM, self).__init__()
        self._env, self._f_se, self._f_ac, \
            self._f_forward, self._f_inverse, self._episode_n, self._learning_rate, \
            self._discount_factor, self._entropy, self._batch_size = \
            env, f_se, f_ac, f_forward, f_inverse, episode_n, learning_rate, \
            discount_factor, entropy, batch_size

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)

        def create_optimizer():
            return tf.train.AdamOptimizer(self._learning_rate)

        def create_agent(n_optimizer, global_step):
            # all ScheduledParam hyper parameters are mutable objects.
            # so we will not want to use same object for different Agent instances.
            entropy = hrl.utils.clone_params(self._entropy)
            agent = hrl.ActorCriticWithICM(
                env=self._env,
                f_se=self._f_se,
                f_ac=self._f_ac,
                f_forward=self._f_forward,
                f_inverse=self._f_inverse,
                state_shape=state_shape,
                # ACUpdate arguments
                discount_factor=self._discount_factor,
                entropy=entropy,
                target_estimator=None,
                max_advantage=100.0,
                # optimizer arguments
                network_optimizer=n_optimizer,
                # sampler arguments
                sampler=None,
                batch_size=self._batch_size,

                global_step=global_step,
            )
            return agent

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.create_session(config=config) as sess:
            agent.set_session(sess)
            logging.warning("render once:%s", args.render_once)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor, max_episode_len=10000,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=args.render_once,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(self._episode_n)


class A3CExperimentWithI2A(Experiment):
    def __init__(self,
                 env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder,
                 # env, f_se_1, f_se_2, f_se_3, f_se_4, f_ac, f_env, f_rollout, f_encoder,
                 episode_n=1000,
                 learning_rate=1e-4,
                 discount_factor=0.9,
                 entropy=1e-2,
                 batch_size=8,
                 policy_with_iaa=False,
                 compute_with_diff=False,
                 with_momentum=True,
                 dynamic_rollout=[1, 3, 5],
                 dynamic_skip_step=[5000, 15000],
                 save_image_interval=1000,
                 with_ob=False
                 ):
        super(A3CExperimentWithI2A, self).__init__()
        self._env, self._f_se, self._f_ac, self._f_tran, self._f_decoder,\
            self._f_rollout, self._f_encoder, self._episode_n, self._learning_rate, \
            self._discount_factor, self._entropy, self._batch_size, \
            self.policy_with_iaa, self.compute_with_diff, self.with_momentum, \
            self.dynamic_rollout, self.dynamic_skip_step, self._save_image_interval, self._with_ob = \
            env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, episode_n, learning_rate, \
            discount_factor, entropy, batch_size, policy_with_iaa, compute_with_diff, with_momentum, dynamic_rollout,\
            dynamic_skip_step, save_image_interval, with_ob

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)

        def create_optimizer():
            return tf.train.AdamOptimizer(self._learning_rate)

        def create_agent(n_optimizer, global_step):
            # all ScheduledParam hyper parameters are mutable objects.
            # so we will not want to use same object for different Agent instances.
            entropy = hrl.utils.clone_params(self._entropy)
            agent = hrl.ActorCriticWithI2A(
                num_action=self._env.action_space.n,
                f_se=self._f_se,
                f_ac=self._f_ac,
                f_tran=self._f_tran,
                f_decoder=self._f_decoder,
                f_rollout=self._f_rollout,
                f_encoder = self._f_encoder,
                state_shape=state_shape,
                # ACUpdate arguments
                discount_factor=self._discount_factor,
                entropy=entropy,
                target_estimator=None,
                max_advantage=100.0,
                # optimizer arguments
                network_optimizer=n_optimizer,
                # sampler arguments
                sampler=None,
                policy_with_iaa=self.policy_with_iaa,
                compute_with_diff=self.compute_with_diff,
                with_momentum=self.with_momentum,
                dynamic_rollout=self.dynamic_rollout,
                dynamic_skip_step=self.dynamic_skip_step,
                batch_size=self._batch_size,
                save_image_interval=self._save_image_interval,
                log_dir=args.logdir,
                with_ob=self._with_ob,
                global_step=global_step,
            )
            return agent

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.create_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor, max_episode_len=10000,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=args.render_once,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(self._episode_n)


class A3CExperimentWithI2AOB(Experiment):
    def __init__(self,
                 env, f_se, f_ac, f_env, f_rollout, f_encoder,
                 # env, f_se_1, f_se_2, f_se_3, f_se_4, f_ac, f_env, f_rollout, f_encoder,
                 episode_n=1000,
                 learning_rate=1e-4,
                 discount_factor=0.9,
                 entropy=1e-2,
                 batch_size=8
                 ):
        super(A3CExperimentWithI2AOB, self).__init__()
        self._env, self._f_se, self._f_ac, self._f_env,\
            self._f_rollout, self._f_encoder, self._episode_n, self._learning_rate, \
            self._discount_factor, self._entropy, self._batch_size = \
            env, f_se, f_ac, f_env, f_rollout, f_encoder, episode_n, learning_rate, \
            discount_factor, entropy, batch_size

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)

        def create_optimizer():
            return tf.train.AdamOptimizer(self._learning_rate)

        def create_agent(n_optimizer, global_step):
            # all ScheduledParam hyper parameters are mutable objects.
            # so we will not want to use same object for different Agent instances.
            entropy = hrl.utils.clone_params(self._entropy)
            agent = hrl.ActorCriticWithI2AOB(
                num_action=self._env.action_space.n,
                f_se=self._f_se,
                # f_se_1=self._f_se_1,
                # f_se_2=self._f_se_2,
                # f_se_3=self._f_se_3,
                # f_se_4=self._f_se_4,
                f_ac=self._f_ac,
                f_env=self._f_env,
                f_rollout=self._f_rollout,
                f_encoder = self._f_encoder,
                state_shape=state_shape,
                # ACUpdate arguments
                discount_factor=self._discount_factor,
                entropy=entropy,
                target_estimator=None,
                max_advantage=100.0,
                # optimizer arguments
                network_optimizer=n_optimizer,
                # sampler arguments
                sampler=None,
                batch_size=self._batch_size,

                global_step=global_step,
            )
            return agent

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.create_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor, max_episode_len=10000,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=True,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(self._episode_n)


class PPOExperiment(Experiment):

    def __init__(self,
                 env, f_create_net, episode_n=1000,
                 discount_factor=0.9,
                 entropy=1e-2,
                 clip_epsilon=0.2,
                 epoch_per_step=4,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                 batch_size=8,
                 horizon=256
                 ):
        super(PPOExperiment, self).__init__()
        self._env, self._f_create_net, self._episode_n, \
            self._discount_factor, self._entropy, self._clip_epsilon, self._epoch_per_step, \
            self._network_optimizer_ctor, self._batch_size, self._horizon = \
            env, f_create_net, episode_n, \
            discount_factor, entropy, clip_epsilon, epoch_per_step, \
            network_optimizer_ctor, batch_size, horizon

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)

        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
        agent = hrl.PPO(
            f_create_net=self._f_create_net,
            state_shape=state_shape,
            # ACUpdate arguments
            discount_factor=self._discount_factor,
            entropy=self._entropy,
            clip_epsilon=self._clip_epsilon,
            epoch_per_step=self._epoch_per_step,
            # optimizer arguments
            network_optimizer=self._network_optimizer_ctor(),
            batch_size=self._batch_size,
            horizon=self._horizon,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir
            )
            runner.episode(self._episode_n)


class OTDQNModelExperiment(Experiment):

    def __init__(self, env, episode_n,
                 f_create_q, f_se, f_transition,
                 f_decoder,
                 lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1,
                 network_optimizer=None,
                 max_gradient=10.0,
                 update_interval=4,
                 replay_size=1024,
                 batch_size=12,
                 with_momentum = True,
                 curriculum=[1, 3, 5],
                 skip_step=[10000, 20000],
                 sampler_creator=None,
                 asynchronous=False,
                 save_image_interval=10000,
                 with_ob=False
                 ):
        super(OTDQNModelExperiment, self).__init__()

        self._env, self._episode_n, \
            self._f_create_q, self._f_se, self._f_transition, \
            self._f_decoder, \
            self._lower_weight, self._upper_weight, \
            self._rollout_depth, self._discount_factor, self._ddqn, self._target_sync_interval, self._target_sync_rate, \
            self._greedy_epsilon, \
            self._network_optimizer, \
            self._max_gradient, \
            self._update_interval, \
            self._replay_size, \
            self._batch_size, \
            self._curriculum, \
            self._skip_step, \
            self._sampler_creator,\
            self._save_image_interval,\
            self._with_ob,\
            self._with_momentum = \
            env, episode_n, \
            f_create_q, f_se, f_transition, \
            f_decoder, \
            lower_weight, upper_weight, \
            rollout_depth, discount_factor, ddqn, target_sync_interval, target_sync_rate, \
            greedy_epsilon, \
            network_optimizer, \
            max_gradient, \
            update_interval, \
            replay_size, \
            batch_size, \
            curriculum, \
            skip_step, \
            sampler_creator, \
            save_image_interval, \
            with_ob, \
            with_momentum
        self._asynchronous = asynchronous

    def run(self, args):
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        sampler = None if self._sampler_creator is None else self._sampler_creator(args)

        agent = ot_model.OTModel(f_create_q=self._f_create_q, f_se=self._f_se, f_transition=self._f_transition,
                        f_decoder=self._f_decoder,
                        lower_weight=self._lower_weight, upper_weight=self._upper_weight,
                        state_shape=self._env.observation_space.shape, num_actions=self._env.action_space.n,
                        rollout_depth=self._rollout_depth, discount_factor=self._discount_factor,
                        ddqn=self._ddqn,
                        target_sync_interval=self._target_sync_interval, target_sync_rate=self._target_sync_rate,
                        greedy_epsilon=self._greedy_epsilon,
                        network_optimizer=self._network_optimizer,
                        max_gradient=self._max_gradient,
                        update_interval=self._update_interval,
                        replay_size=self._replay_size,
                        batch_size=self._batch_size,
                        sampler=sampler,
                        with_momentum=self._with_momentum,
                        curriculum=self._curriculum,
                        skip_step=self._skip_step,
                        save_image_interval=self._save_image_interval,
                        log_dir=args.logdir,
                        with_ob=self._with_ob,
                        global_step=global_step)
        if self._asynchronous:
            agent = AsynchronousAgent(agent=agent, method='ratio', rate=6.0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=args.render_once,
            )
            runner.episode(self._episode_n)
        super(OTDQNModelExperiment, self).run(args)