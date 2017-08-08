# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import unittest

import tensorflow as tf
import gym
import hobotrl as hrl


class TestAC(unittest.TestCase):
    def test_run(self):
        tf.reset_default_graph()
        env = gym.make('Pendulum-v0')
        env = hrl.envs.ScaledRewards(env, 0.1)
        state_shape = list(env.observation_space.shape)
        dim_action = env.action_space.shape[0]
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

        def f_net(inputs):
            l2 = 1e-8
            state, action = inputs[0], inputs[1]
            actor = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action, activation_out=tf.nn.tanh, l2=l2, var_scope="action")
            se = hrl.network.Utils.layer_fcs(state, [200], 100, activation_out=None, l2=l2, var_scope="se")
            se = tf.concat([se, action], axis=-1)
            q = hrl.network.Utils.layer_fcs(se, [100], 1, activation_out=None, l2=l2, var_scope="q")
            q = tf.squeeze(q, axis=1)
            return {"q": q, "action": actor}

        sampler = hrl.async.AsyncTransitionSampler(hrl.playback.MapPlayback(1000), 32)
        agent = hrl.DPG(
            f_create_net=f_net, state_shape=state_shape, dim_action=dim_action,
            # ACUpdate arguments
            discount_factor=0.9, target_estimator=None,
            # optimizer arguments
            network_optimizer=hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
            max_gradient=10.0,
            # policy arguments
            ou_params=(0, 0.2, 0.2),
            # target network sync arguments
            target_sync_interval=10,
            target_sync_rate=0.01,
            # sampler arguments
            sampler=sampler,
            batch_size=32,
            global_step=global_step,
        )
        agent = hrl.async.AsynchronousAgent(agent)
        # another type of
        # def create_agent():
        #     return DPG(
        #         f_create_net=f_net, state_shape=state_shape, dim_action=dim_action,
        #         # ACUpdate arguments
        #         discount_factor=0.9, target_estimator=None,
        #         # optimizer arguments
        #         network_optimizer=network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
        #         max_gradient=10.0,
        #         # policy arguments
        #         ou_params=(0, 0.2, 0.2),
        #         # target network sync arguments
        #         target_sync_interval=10,
        #         target_sync_rate=0.01,
        #         # sampler arguments
        #         sampler=sampler,
        #         batch_size=32,
        #         global_step=global_step,
        #     )
        #
        # agent = async.AsynchronousAgent2(create_agent)

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
            agent.stop()
