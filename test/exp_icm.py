import sys
sys.path.append(".")
import logging
import math
import numpy as np
import gym
import cv2
import matplotlib.colors as colors
from exp_algorithms import *
import hobotrl.environments as envs


class RewardSparseCartPole(gym.Wrapper):

    def __init__(self, env):
        super(RewardSparseCartPole, self).__init__(env)
        self.observation_space.high[2] = 3 * 2 * math.pi / 360
        self.observation_space.low[2] = -3 * 2 * math.pi / 360

    def _step(self, action):
        self.step_count = 0
        observation, reward, done, info = self.env.step(action)
        if not done:
            self.step_count += 1
        if done:
            self.step_count = 0
        # if not -0.1 < observation[2] < 0.:
        #     reward = 0
        if self.step_count == 8:
            reward = 1
        else:
            reward = 0
        print "-----------------reward: %s, step count: %s", reward, self.step_count
        return observation, reward, done, info


class A3CCartPoleWithICM(A3CExperimentWithICM):
    def __init__(self, env=None, f_se=None, f_ac=None, f_forward=None, f_inverse=None, episode_n=1000,
                 learning_rate=5e-5, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-4, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('CartPole-v0')
            # env = RewardSparseCartPole(env)

        if f_forward is None:
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
                print "-------------dim action: %s, \npi: %s", dim_action, pi
                pi = tf.squeeze(pi, axis=0)

                # critic
                v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=2)
                v = tf.squeeze(v, axis=0)

                return {"pi": pi, "v": v}

            def create_forward(inputs):
                l2 = 1e-7
                action_sample = inputs[0]
                phi1 = inputs[1]

                # forward model
                f = tf.concat([phi1, action_sample], 1)
                phi2_hat = hrl.utils.Network.layer_fcs(f, [200], 200,
                                                       activation_hidden=tf.nn.relu,
                                                       activation_out=tf.nn.relu,
                                                       l2=l2,
                                                       var_scope="next_state_predict")
                return {"phi2_hat": phi2_hat}

            def create_inverse(inputs):
                l2 = 1e-7
                phi1 = inputs[0]
                phi2 = inputs[1]

                # inverse model
                g = tf.concat([phi1, phi2], 1)
                logits = hrl.utils.Network.layer_fcs(g, [200], dim_action,
                                                     activation_hidden=tf.nn.relu,
                                                     activation_out=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="action_predict")

                return {"logits": logits}

            f_ac = create_ac
            f_se = create_se
            f_inverse = create_inverse
            f_forward = create_forward

        super(A3CCartPoleWithICM, self).__init__(env, f_se, f_ac, f_forward, f_inverse, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)



Experiment.register(A3CCartPoleWithICM, "A3C with ICM for CartPole")


if __name__ == '__main__':
    Experiment.main()