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
import playground.optimal_tighten as play_ot
import hobotrl.algorithms.ot as ot
import playground.a3c_onoff as a3coo


def state_trans_atari(state):
    gray = np.asarray(np.dot(state, [0.299, 0.587, 0.114]))
    gray = cv2.resize(gray, (84, 84))
    return np.asarray(gray.reshape(gray.shape + (1,)), dtype=np.uint8)


def f_net_atari(inputs, num_action, is_training):
    input_var = inputs
    print "input size:", input_var
    out = hrl.utils.Network.conv2ds(input_var, shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], out_flatten=True,
                                    activation=tf.nn.relu,
                                    l2=1e-8, var_scope="convolution")
    out = hrl.utils.Network.layer_fcs(input_var=out, shape=[256], out_count=num_action,
                                      activation_hidden=tf.nn.relu,
                                      activation_out=None,
                                      l2=1e-8, var_scope="fc")
    return out


class DQNPong(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """
    def run(self, args):
        reward_decay = 0.99
        batch_size = 32
        replay_size = 10000
        epsilon = hrl.utils.CappedLinear(step=4e5, start=1.0, end=0.01)
        # epsilon = hrl.utils.CosSequence(step=4e5, start=1.0, end=0.05)
        env = gym.make("PongNoFrameskip-v4")

        # env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.5,
        #                                  reward_shaping_proc=hrl.envs.InfoChange(decrement_weight={"ale.lives": -2}),
        #                                  state_augment_proc=state_trans_atari, state_stack_n=4, state_skip=4)

        env = ScaledFloatFrame(wrap_dqn(env))

        state_shape = [84, 84, 4]  # list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=epsilon,
            # DeepQFuncMixin params
            gamma=reward_decay,
            f_net_dqn=f_net_atari, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=(tf.train.AdamOptimizer(learning_rate=0.0001), 1.0),
            schedule=(4, 1000),
            greedy_policy=True,
            ddqn=True,
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
            buffer_param_dict={
                "capacity": replay_size,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (),
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                },
                # "augment_offset": {
                #     "state": -0,
                #     "next_state": -0,
                # },
                # "augment_scale": {
                #     "state": 1/255.0,
                #     "next_state": 1/255.0,
                # }
            },
            batch_size=batch_size,
            global_step=global_step
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=sys.maxint, logdir=args.logdir)
            runner.episode(100000)

Experiment.register(DQNPong, "DQN for Pong")


class PERPong(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """
    def run(self, args):
        reward_decay = 0.99
        batch_size = 32
        replay_size = 10000
        epsilon = hrl.utils.CappedLinearSequence(step=3e5, start=1.0, end=0.05)
        # epsilon = hrl.utils.CosSequence(step=3e5, start=0.95, end=0.05)

        env = gym.make("PongNoFrameskip-v4")
        # env = hrl.envs.C2DEnvWrapper(env, [5])

        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.5,
                                         reward_shaping_proc=hrl.envs.InfoChange(decrement_weight={"ale.lives": -2}),
                                         state_augment_proc=state_trans_atari, state_stack_n=4)

        state_shape = [84, 84, 4]  # list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = per.PrioritizedDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=epsilon,
            # DeepQFuncMixin params
            gamma=reward_decay,
            f_net_dqn=f_net_atari, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=(tf.train.AdamOptimizer(learning_rate=0.0001), 1.0),
            schedule=(4, 1000),
            greedy_policy=True,
            ddqn=True,
            # ReplayMixin params
            buffer_class=hrl.playback.NearPrioritizedPlayback,
            buffer_param_dict={
                "capacity": replay_size,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (),
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                },
                "priority_bias": 0.5,
                "importance_weight": hrl.utils.CappedLinearSequence(1e6, 0.4, 1.0),
                "augment_offset": {
                    "state": -0,
                    "next_state": -0,
                },
                "augment_scale": {
                    "state": 1 / 255.0,
                    "next_state": 1 / 255.0,
                }
            },
            batch_size=batch_size,
            global_step=global_step,
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=0,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=sys.maxint, logdir=args.logdir)
            runner.episode(100000)

Experiment.register(PERPong, "PERDQN for Pong")


class ACOOPong(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """
    def run(self, args):
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
            reward_decay = 0.99
            on_batch_size = 32
            off_batch_size = 32
            replay_size = 10000
            prob_min = 5e-3
            # entropy = 1e-3
            # entropy = hrl.utils.CosSequence(1e6 / on_batch_size, 1e-5, 1e-7)
            entropy = 1e-2
            l2 = 1e-8
            optimizer = lambda: tf.train.AdamOptimizer(1e-4)  # called later on correct device context
            kwargs = {"ddqn": False, "aux_r": False, "aux_d": False, "reward_decay": reward_decay}
            env = gym.make("PongNoFrameskip-v4")

            env = ScaledFloatFrame(wrap_dqn(env))

            def create_ac_atari(input_state, num_action, **kwargs):
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se")

                q = hrl.utils.Network.layer_fcs(se, [256], num_action,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="q")
                pi = hrl.utils.Network.layer_fcs(se, [256], num_action,
                                                 activation_hidden=tf.nn.relu,
                                                 # activation_out=tf.nn.softplus,
                                                 l2=l2,
                                                 var_scope="pi")

                pi = tf.nn.softmax(pi)
                # pi = pi + prob_min
                # pi = pi / tf.reduce_sum(pi, axis=-1, keep_dims=True)
                r = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="r")

                return {"pi": pi, "q": q, "se": se, "r": r}

            def create_duel_ac_atari(input_state, num_action, **kwargs):
                with tf.variable_scope("se"):
                    se = hrl.utils.Network.conv2ds(input_state,
                                                   shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                                   out_flatten=True,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="convolution")
                    se = hrl.utils.Network.layer_fcs(se, [], 256, activation_hidden=None,
                                                     activation_out=tf.nn.relu, l2=l2, var_scope="fc")

                    a = hrl.utils.Network.layer_fcs(se, [], num_action, activation_hidden=None,
                                                    activation_out=None, l2=l2, var_scope="a")
                    a = a - tf.reduce_mean(a, axis=-1, keep_dims=True)

                with tf.variable_scope("q"):
                    v = hrl.utils.Network.layer_fcs(se, [], 1,
                                                    activation_hidden=tf.nn.relu,
                                                    l2=l2,
                                                    var_scope="v")
                    q = v + a

                pi = hrl.utils.Network.layer_fcs(a, [], num_action,
                                                 activation_hidden=tf.nn.relu,
                                                 # activation_out=tf.nn.softplus,
                                                 l2=l2,
                                                 var_scope="pi")

                pi = tf.nn.softmax(pi)
                # pi = pi + prob_min
                # pi = pi / tf.reduce_sum(pi, axis=-1, keep_dims=True)
                r = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="r")

                return {"pi": pi, "q": q, "se": se, "r": r}

            state_shape = [84, 84, 4]  # list(env.observation_space.shape)
            with tf.device("/job:worker/task:0"):
                global_step = tf.get_variable('global_step', [],
                                              initializer=tf.constant_initializer(0),
                                              trainable=False)
                global_net = a3coo.ActorCritic(0, "global_net", state_shape, env.action_space.n,
                                               create_duel_ac_atari, optimizer=optimizer(),
                                               global_step=global_step, **kwargs)

            for i in range(worker_n):
                with tf.device("/job:worker/task:%d" % i):
                    worker = a3coo.A3CAgent(
                        index=i,
                        parent_net=global_net,
                        create_net=create_duel_ac_atari,
                        state_shape=state_shape,
                        num_actions=env.action_space.n,
                        replay_capacity=replay_size,
                        train_on_interval=on_batch_size,
                        train_off_interval=0,
                        target_follow_interval=0,
                        off_batch_size=off_batch_size,
                        entropy=entropy,
                        global_step=global_step,
                        optimizer=optimizer(),
                        **kwargs
                    )
                    if i == args.index:
                        agent = worker

            sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=args.index,
                                       init_op=tf.global_variables_initializer(), save_dir=args.logdir)

            with sv.prepare_or_wait_for_session(server.target) as sess:
                agent.set_session(sess)
                runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                            evaluate_interval=sys.maxint, render_interval=sys.maxint,
                                            logdir=args.logdir if args.index == 0 else None)
                runner.episode(100000)

Experiment.register(ACOOPong, "Actor Critic for Pong")


class DQNBattleZone(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    """
    def run(self, args):
        reward_decay = 0.99
        batch_size = 32
        replay_size = 10000
        epsilon = hrl.utils.CappedLinearSequence(step=4e5, start=1.0, end=0.01)
        # epsilon = hrl.utils.CosSequence(step=4e5, start=1.0, end=0.05)
        env = gym.make("BattleZoneNoFrameskip-v4")

        # env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.5,
        #                                  reward_shaping_proc=hrl.envs.InfoChange(decrement_weight={"ale.lives": -2}),
        #                                  state_augment_proc=state_trans_atari, state_stack_n=4, state_skip=4)

        env = ScaledFloatFrame(wrap_dqn(env))

        state_shape = [84, 84, 4]  # list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = dqn.DQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=epsilon,
            # DeepQFuncMixin params
            gamma=reward_decay,
            f_net_dqn=f_net_atari, state_shape=state_shape, num_actions=env.action_space.n,
            training_params=(tf.train.AdamOptimizer(learning_rate=0.0001), 1.0),
            schedule=(4, 1000),
            greedy_policy=True,
            ddqn=True,
            # ReplayMixin params
            buffer_class=hrl.playback.MapPlayback,
            buffer_param_dict={
                "capacity": replay_size,
                "sample_shapes": {
                    'state': state_shape,
                    'action': (),
                    'reward': (),
                    'next_state': state_shape,
                    'episode_done': ()
                },
                # "augment_offset": {
                #     "state": -0,
                #     "next_state": -0,
                # },
                # "augment_scale": {
                #     "state": 1/255.0,
                #     "next_state": 1/255.0,
                # }
            },
            batch_size=batch_size,
            global_step=global_step
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(graph=tf.get_default_graph(), worker_index=args.index,
                                   init_op=tf.global_variables_initializer(), save_dir=args.logdir)
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=sys.maxint,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(100000)

Experiment.register(DQNBattleZone, "DQN for BattleZone")


if __name__ == '__main__':
    Experiment.main()
