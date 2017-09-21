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


class ICMLinear(A3CExperimentWithICM):
    def __init__(self, env=None, f_se=None, f_ac=None, f_forward=None, f_inverse=None, episode_n=10000,
                 learning_rate=1e-4, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('MountainCar-v0')
            env._max_episode_steps = 10000
            # env = envs.AugmentEnvWrapper(env, reward_scale=0.01)
            # env = BalanceRewardAcrobot(env)
            # env = gym.wrappers.Monitor(env, "./log/AcrobotNew/ICMMaxlen200", force=True)

        if (f_forward and f_se and f_inverse and f_ac) is None:
            dim_action = env.action_space.n

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
                pi = hrl.utils.Network.layer_fcs(se, [256], dim_action,
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
                logits = hrl.utils.Network.layer_fcs(g, [200], dim_action,
                                                     activation_hidden=tf.nn.elu,
                                                     # activation_out=tf.nn.relu,
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


if __name__ == '__main__':
    Experiment.main()
