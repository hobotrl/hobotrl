# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import unittest

import tensorflow as tf
import gym
import hobotrl as hrl


class TestICM(unittest.TestCase):
    def test_run(self):
        tf.reset_default_graph()
        env = gym.make('Acrobot-v1')
        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

        def create_se(inputs):
            l2 = 1e-7
            input_state = inputs[0]

            se = hrl.utils.Network.layer_fcs(input_state, [200], 200,
                                             activation_hidden=tf.nn.elu,
                                             activation_out=tf.nn.elu,
                                             l2=l2,
                                             var_scope="se")
            return {"se": se}

        def create_ac(inputs):
            l2 = 1e-7
            se = inputs[0]

            # actor
            pi = hrl.utils.Network.layer_fcs(se, [256], 3,
                                             activation_hidden=tf.nn.elu,
                                             activation_out=tf.nn.softmax,
                                             l2=l2,
                                             var_scope="pi")

            # critic
            v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                            activation_hidden=tf.nn.elu,
                                            l2=l2,
                                            var_scope="v")
            v = tf.squeeze(v, axis=1)
            return {"pi": pi, "v": v}

        def create_forward(inputs):
            l2 = 1e-7
            action_sample = inputs[0]
            phi1 = inputs[1]

            # forward model
            f = tf.concat([phi1, action_sample], 1)
            phi2_hat = hrl.utils.Network.layer_fcs(f, [200], 200,
                                                   activation_hidden=tf.nn.elu,
                                                   activation_out=tf.nn.elu,
                                                   l2=l2,
                                                   var_scope="next_state_predict")
            return {"phi2_hat": phi2_hat}

        def create_inverse(inputs):
            l2 = 1e-7
            phi1 = inputs[0]
            phi2 = inputs[1]

            # inverse model
            g = tf.concat([phi1, phi2], 1)
            logits = hrl.utils.Network.layer_fcs(g, [200], 3,
                                                 activation_hidden=tf.nn.elu,
                                                 # activation_out=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="action_predict")
            return {"logits": logits}

        f_ac = create_ac
        f_se = create_se
        f_inverse = create_inverse
        f_forward = create_forward

        agent = hrl.ActorCriticWithICM(
            env=env,
            f_se=f_se,
            f_ac=f_ac,
            f_forward=f_forward,
            f_inverse=f_inverse,
            state_shape=state_shape,
            # ACUpdate arguments
            discount_factor=0.9,
            entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4),
            target_estimator=None,
            max_advantage=100.0,
            # optimizer arguments
            network_optimizer=hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4), grad_clip=10.0),
            max_gradient=10.0,
            # sampler arguments
            sampler=None,
            batch_size=8,
            global_step=global_step,
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with agent.create_session(config=config, save_dir=None) as sess:
            runner = hrl.EnvRunner(
                env, agent, evaluate_interval=sys.maxint,
                render_interval=sys.maxint, logdir=None
            )
            runner.episode(50)