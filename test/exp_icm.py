import sys
sys.path.append(".")
import logging
import numpy as np
import gym
import cv2
import matplotlib.colors as colors
from exp_algorithms import *
import hobotrl.environments as envs


class RewardSparsePendulum(gym.Wrapper):

    def __init__(self, env):
        super(RewardSparsePendulum, self).__init__(env)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        pass

SE = Network(s_t,f_se)

pi = Network(se.output(), f_pi)
SE_t = SE(s_t)
SE_t_1 = SE(s_t1)
im = Network([SE_t.outptu(),SE_t1.outptu()], f_im)
fm = Network([SE_t.output(), action], f_fm)
Function

class A3CPendulumWithICM(A3CExperimentWithICM):
    def __init__(self, env=None, f_se=None, f_pi=None, f_v=None, f_forward=None, f_inverse=None, episode_n=1000,
                 learning_rate=5e-5, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-4, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('Pendulum-v0')
            env = RewardSparsePendulum(env)

        if f_forward and f_inverse and f_pi and f_se and f_v is None:
            dim_action = env.action_space.n

            def create_se(input_state):
                l2 = 1e-7

                se = hrl.utils.Network.layer_fcs(input_state, [200], 200,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="se")
                return {"se": se}

            def create_ac(se):
                l2 = 1e-7

                # actor
                pi = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                # critic
                v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)

                return {"pi": pi, "v": v}

            def create_forward(action_sample, phi1):
                l2 = 1e-7

                # forward model
                f = tf.concat([phi1, action_sample], 1)
                phi2_hat = hrl.utils.Network.layer_fcs(f, [200], 200,
                                                     activation_hidden=tf.nn.relu,
                                                     activation_out=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="next state predict")
                return {"phi2_hat": phi2_hat}

            def create_inverse(phi1, phi2):
                l2 = 1e-7

                # inverse model
                g = tf.concat([phi1, phi2], 1)
                logits = hrl.utils.Network.layer_fcs(g, [200], dim_action,
                                                     activation_hidden=tf.nn.relu,
                                                     activation_out=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="action predict")

                return {"logits": logits}

            f_ac = create_ac
            f_se = create_se
            f_inverse = create_inverse
            f_forward = create_forward

        super(A3CPendulumWithICM, self).__init__(env, f_se, f_ac, f_forward, f_inverse, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)

Experiment.register(A3CPendulumWithICM, "continuous A3C for CarRacing")