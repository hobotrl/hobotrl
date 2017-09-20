# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import unittest

import tensorflow as tf
import gym
import hobotrl as hrl


class TestPPO(unittest.TestCase):
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

        def f_net(inputs):
            l2 = 1e-4
            state = inputs[0]
            q = hrl.network.Utils.layer_fcs(state, [200, 100], env.action_space.n,
                                            l2=l2, var_scope="q")
            pi = hrl.network.Utils.layer_fcs(state, [200, 100], env.action_space.n,
                                             activation_out=tf.nn.softmax, l2=l2, var_scope="pi")
            return {"q": q, "pi": pi}

        agent = hrl.PPO(
            f_create_net=f_net,
            state_shape=state_shape,
            discount_factor=0.9,
            entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-2),
            clip_epsilon=0.2,
            epoch_per_step=4,
            batch_size=16,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config) as sess:
            runner = hrl.EnvRunner(
                env, agent, evaluate_interval=sys.maxint,
                render_interval=sys.maxint, logdir=None
            )
            runner.episode(50)


if __name__ == '__main__':
    unittest.main()