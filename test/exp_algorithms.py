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
import hobotrl.sampling as sampling
import hobotrl.algorithms.ac as ac
import hobotrl.algorithms.dqn as dqn
import hobotrl.algorithms.dpg as dpg
import hobotrl.algorithms.ot as ot
import playground.a3c_onoff as a3coo
import playground.a3c_continuous_onoff as a3ccoo


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
            greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.5, 0.1),
            network_optimizer=self._network_optimizer_ctor(),
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
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir,
                render_once=True,
            )
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
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
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

        super(OTDQNExperiment, self).__init__()

    def run(self, args):

        state_shape = list(self._env.observation_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
             initializer=tf.constant_initializer(0), trainable=False
        )
        agent = hrl.algorithms.ot.DQN(
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
            greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.5, 0.1),
            network_optmizer=self._network_optimizer_ctor(),
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
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=sys.maxint, logdir=args.logdir,
                render_once=True,
            )
            runner.episode(self._episode_n)


class DPGExperiment(Experiment):

    def __init__(self, env,
                 f_create_net,
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
        self._env, self._f_create_net, self._episode_n,\
            self._discount_factor, self._network_optimizer_ctor, \
            self._ou_params, self._target_sync_interval, self._target_sync_rate, \
            self._batch_size, self._replay_capacity = \
            env, f_create_net, episode_n, \
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
            f_create_net=self._f_create_net,
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
        sv = agent.init_supervisor(
            graph=tf.get_default_graph(), worker_index=0,
            init_op=tf.global_variables_initializer(), save_dir=args.logdir
        )
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
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

            sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=args.index,
                                       init_op=tf.global_variables_initializer(), save_dir=args.logdir)
            with sv.prepare_or_wait_for_session(server.target) as sess:
                agent.set_session(sess)
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

            sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=args.index,
                                       init_op=tf.global_variables_initializer(), save_dir=args.logdir)
            with sv.prepare_or_wait_for_session(server.target) as sess:
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
        sv = agent.init_supervisor(
            graph=tf.get_default_graph(), worker_index=0,
            init_op=tf.global_variables_initializer(), save_dir=args.logdir
        )
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
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

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.wait_for_session() as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=True,
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

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.wait_for_session() as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(self._env, agent, reward_decay=self._discount_factor,
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
        sv = agent.init_supervisor(
            graph=tf.get_default_graph(), worker_index=0,
            init_op=tf.global_variables_initializer(), save_dir=args.logdir
        )
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(
                self._env, agent, evaluate_interval=sys.maxint,
                render_interval=args.render_interval, logdir=args.logdir
            )
            runner.episode(self._episode_n)