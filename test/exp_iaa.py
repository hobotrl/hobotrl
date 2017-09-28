import sys
sys.path.append(".")
import logging
import math
import numpy as np
import gym
from gym import spaces
import cv2
import matplotlib.colors as colors
from exp_algorithms import *
import hobotrl.environments as envs


class GoTransposeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GoTransposeWrapper, self).__init__(env)
        self.observation_space = spaces.Box(0, 1, [9, 9, 3])

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.transpose(observation, (2, 1, 0))
        print "----------------------------", np.shape(observation)
        return observation, reward, done, info

    def _reset(self):
        observation = self.env.reset()
        observation = np.transpose(observation, (2, 1, 0))
        print "----------------------------", np.shape(observation)
        return observation


class I2A(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_env=None, f_rollout=None, f_encoder = None, episode_n=10000,
                 learning_rate=1e-4, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('Go9x9-v0')
            env = GoTransposeWrapper(env)
            # env._max_episode_steps = 10000
            # env = envs.FrameStack(env, k=4)
            # env = gym.wrappers.Monitor(env, "./log/AcrobotNew/ICMMaxlen200", force=True)

        if (f_env and f_rollout and f_ac) is None:
            dim_action = env.action_space.n
            dim_observation = env.observation_space.shape

            def create_se(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se_conv")
                se_linear = hrl.utils.Network.layer_fcs(se_conv, [], 200,
                                                activation_hidden=tf.nn.relu,
                                                activation_out=tf.nn.relu,
                                                l2=l2,
                                                var_scope="se_linear")
                return {"se": se_linear}

            def create_ac(inputs):
                l2 = 1e-7
                input_feature = inputs[0]

                v = hrl.utils.Network.layer_fcs(input_feature, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(input_feature, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                return {"v": v, "pi": pi}

            def create_rollout(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                # rollout that imitates the A3C policy
                rollout_se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="rollout_se")

                rollout_action = hrl.utils.Network.layer_fcs(rollout_se, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")
                return {"rollout_action": rollout_action}

            def create_env(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_state = tf.squeeze(tf.stack(input_state), axis=0)
                input_action = inputs[1]
                input_action = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                                                      [dim_observation[0], dim_observation[1]])
                full_input = tf.concat([input_action, input_state], axis=3)

                conv_1 = hrl.utils.Network.conv2ds(full_input,
                                               shape=[(32, 8, 4)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="conv_1")

                up_1 = tf.image.resize_images(conv_1, [dim_observation[0], dim_observation[1]])

                concat_1 = tf.concat([up_1, input_state], axis=3)

                conv_2 = hrl.utils.Network.conv2ds(concat_1,
                                                   shape=[(16, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")

                up_2 = tf.image.resize_images(conv_2, [dim_observation[0], dim_observation[1]])

                concat_2 = tf.concat([up_2, input_state], axis=3)

                conv_3 = hrl.utils.Network.conv2ds(concat_2,
                                                   shape=[(8, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_3")

                conv_r = hrl.utils.Network.conv2ds(conv_3,
                                                   shape=[(8, 3, 1)],
                                                   out_flatten=True,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_r")

                reward = hrl.utils.Network.layer_fcs(conv_r, [256], 1,
                                                 activation_hidden=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                up_3 = tf.image.resize_images(conv_3, [dim_observation[0], dim_observation[1]])

                concat_3 = tf.concat([up_3, input_state], axis=3)

                conv_4 = hrl.utils.Network.conv2ds(concat_3,
                                                   shape=[(3, 1, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_4")

                return {"next_state": conv_4, "reward": reward}

            def create_encoder(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_reward = inputs[1]
                print "-------------------------------------"
                print input_state, "\n", input_reward

                rse = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="rse")

                re_conv = hrl.utils.Network.layer_fcs(rse, [], 200,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=tf.nn.relu,
                                            l2=l2,
                                            var_scope="re_conv")

                # re_conv = tf.concat([re_conv, tf.reshape(input_reward, [-1, 1])], axis=1)
                re_conv = tf.concat([re_conv, input_reward], axis=1)

                re = hrl.utils.Network.layer_fcs(re_conv, [], 200,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=tf.nn.relu,
                                            l2=l2,
                                            var_scope="re")

                return {"re": re}

            f_se = create_se
            f_ac = create_ac
            f_env = create_env
            f_rollout = create_rollout
            f_encoder = create_encoder

        super(I2A, self).__init__(env, f_se, f_ac, f_env, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)


Experiment.register(I2A, "A3C with I2A for complex observation state experiments")


if __name__ == '__main__':
    Experiment.main()
