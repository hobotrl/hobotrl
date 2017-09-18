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
        self.env = env
        super(RewardSparseCartPole, self).__init__(env)
        self.env.observation_space.high[2] = 0.1 * math.pi / 360
        self.env.observation_space.low[2] = -0.1 * math.pi / 360
        # self.step_count = 0

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        # if not -0.1 * math.pi / 360 < observation[2] < 0.1 * math.pi / 360:
        #     reward = 0
        # print "-----------------reward: ", reward
        return observation, reward, done, info


class RewardSparseAcrobot(gym.Wrapper):

    def __init__(self, env):
        self.env = env
        super(RewardSparseAcrobot ,self).__init__(env)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = 1.0 if done else 0.0

        return observation, reward, done, info


class BalanceRewardAcrobot(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        super(BalanceRewardAcrobot, self).__init__(env)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewardx = (-np.cos(observation[0]) - np.cos(observation[1] + observation[0]))  ##Swung height is calculated
        if self.rewardx < .5:
            reward = -1.
            self.steps_beyond_done = 0
        if (self.rewardx > .5 and self.rewardx < .8):
            reward = -0.8
            self.steps_beyond_done = 0
        if self.rewardx > .8:
            reward = -0.6
        if self.rewardx > 1:
            reward = -0.4
            self.steps_beyond_done += 1
        if self.steps_beyond_done > 4:
            reward = -0.2
        if self.steps_beyond_done > 8:
            reward = -0.1
        if self.steps_beyond_done > 12:
            reward = 0.

        return observation, reward, self.steps_beyond_done > 20, info


class ICMLinear(A3CExperimentWithICM):
    def __init__(self, env=None, f_se=None, f_ac=None, f_forward=None, f_inverse=None, episode_n=10000,
                 learning_rate=5e-5, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-4, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('Acrobot-v1')
            env = BalanceRewardAcrobot(env)
            env = gym.wrappers.Monitor(env, "/mnt/d/hobotrl/log/Acrobot/AcrobotNewWithICM/")

        if (f_forward and f_se and f_inverse and f_ac) is None:
            dim_action = env.action_space.n

            def create_se(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                se = hrl.utils.Network.layer_fcs(input_state, [200], 200,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="se")
                return {"se": se}

            def create_ac(inputs):
                l2 = 1e-7
                se = inputs[0]

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

        super(ICMLinear, self).__init__(env, f_se, f_ac, f_forward, f_inverse, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)


Experiment.register(ICMLinear, "A3C with ICM for simple observation state experiments")


class A3C(A3CExperiment):
     def __init__(self, env=None, f_create_net=None, episode_n=50000,
                  learning_rate=5e-5, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-4, 1e-4),
                  batch_size=32):
         if env is None:
             env = gym.make('Acrobot-v1')
             env = gym.wrappers.Monitor(env, "/home/qrh/hobotrl/log/ICM/AcrobotOriginA3C/")

         if f_create_net is None:
             dim_action = env.action_space.n

             def create_ac(inputs):
                 l2 = 1e-7
                 input_state = inputs[0]

                 se = hrl.utils.Network.layer_fcs(input_state, [200], 200,
                                                  activation_hidden=tf.nn.relu,
                                                  activation_out=tf.nn.relu,
                                                  l2=l2,
                                                  var_scope="se")
                 v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                 activation_hidden=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="v")
                 v = tf.squeeze(v, axis=1)
                 pi = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                  activation_hidden=tf.nn.relu,
                                                  activation_out=tf.nn.softmax,
                                                  l2=l2,
                                                  var_scope="pi")

                 return {"v": v, "pi": pi}

             f_create_net = create_ac

         super(A3C, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy, batch_size)


Experiment.register(A3C, "A3C without ICM for simple observation state experiments")


if __name__ == '__main__':
    Experiment.main()
