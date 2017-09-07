# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import unittest

import tensorflow as tf
import gym
import hobotrl as hrl


class TestDQN(unittest.TestCase):

    def test_run(self):
        tf.reset_default_graph()
        env = gym.make('Pendulum-v0')
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.ScaledRewards(env, 0.1)
        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

        def f_q(inputs):
            q = hrl.network.Utils.layer_fcs(inputs[0], [200, 100], env.action_space.n, l2=1e-4)
            return {"q": q}

        agent = hrl.DQN(
            f_create_q=f_q,
            state_shape=state_shape,
            # OneStepTD arguments
            num_actions=env.action_space.n,
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
            greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.5, 0.1),
            global_step=global_step,
            network_optimizer=hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0)
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sv = agent.init_supervisor(
            graph=tf.get_default_graph(), worker_index=0,
            init_op=tf.global_variables_initializer(), save_dir=None
        )
        with sv.managed_session(config=config) as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(
                env, agent, evaluate_interval=sys.maxint,
                render_interval=sys.maxint, logdir=None
            )
            runner.episode(50)