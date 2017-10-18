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
        return observation, reward, done, info

    def _reset(self):
        observation = self.env.reset()
        observation = np.transpose(observation, (2, 1, 0))
        return observation


class I2A(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_env=None, f_rollout=None, f_encoder = None, episode_n=10000,
                 learning_rate=1e-4, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = envs.ScaledRewards(env, 0.1)
            env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
            env = envs.FrameStack(env, k=4)
            # env = wrap_car(env, 3, 3, frame=4)

        if (f_env and f_rollout and f_ac) is None:
            dim_action = env.action_space.n
            dim_observation = env.observation_space.shape

            def create_se_1(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(64, 8, 4)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se_conv")
                return {"se_1": se_conv}

            def create_se_2(inputs):
                l2 = 1e-7
                input_feature = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_feature,
                                               shape=[(64, 4, 2)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se_conv")
                return {"se_2": se_conv}

            def create_se_3(inputs):
                l2 = 1e-7
                input_feature = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_feature,
                                               shape=[(64, 4, 2)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se_conv")
                return {"se_3": se_conv}

            def create_se_4(inputs):
                l2 = 1e-7
                input_feature = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_feature,
                                               shape=[(64, 3, 1)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se_conv")
                return {"se_4": se_conv}

            def create_ac(inputs):
                l2 = 1e-7
                input_feature = inputs[0]

                ac_feature = hrl.utils.Network.conv2ds(input_feature,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=True,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_ac")
                se_linear = hrl.utils.Network.layer_fcs(ac_feature, [], 200,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="se_linear")

                v = hrl.utils.Network.layer_fcs(se_linear, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(se_linear, [256], dim_action,
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

            def create_env_deconv(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_state = tf.squeeze(tf.stack(input_state), axis=0)
                input_action = inputs[1]
                input_action = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                                                      [dim_observation[0], dim_observation[1]])
                full_input = tf.concat([input_action, input_state], axis=3)

                conv_1 = hrl.utils.Network.conv2ds(full_input,
                                               shape=[(32, 8, 8), (32, 3, 1)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="conv_1")

                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")

                conv_middle = conv_1 + conv_2

                # reward
                conv_r1 = hrl.utils.Network.conv2ds(conv_middle,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_r1")

                pool_r1 = tf.layers.max_pooling2d(conv_r1, 2, 1)

                pool_r1 = tf.nn.relu(pool_r1)

                conv_r2 = hrl.utils.Network.conv2ds(pool_r1,
                                                    shape=[(32, 3, 1)],
                                                    out_flatten=False,
                                                    activation=tf.nn.relu,
                                                    l2=l2,
                                                    var_scope="conv_r2")

                pool_r2 = tf.layers.max_pooling2d(conv_r2, 2, 1)

                pool_r2 = tf.nn.relu(pool_r2)

                pool_r2 = tf.contrib.layers.flatten(pool_r2)

                reward = hrl.utils.Network.layer_fcs(pool_r2, [256], 1,
                                                     activation_hidden=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_state
                next_state = hrl.utils.Network.deconv2ds(conv_middle,
                                                         shape=[(3, 8, 8)],
                                                         out_flatten=False,
                                                         activation=tf.nn.relu,
                                                         l2=l2,
                                                         var_scope="next_state")

                return {"next_state": next_state, "reward": reward}

            def create_env_upsample(inputs):
                l2 = 1e-7
                # input_state = inputs[0]
                # input_state = tf.squeeze(tf.stack(input_state), axis=0)
                input_se_1 = inputs[0]

                input_se_1 = tf.squeeze(tf.stack(input_se_1), axis=0)

                input_se_2 = inputs[1]
                input_se_2 = tf.squeeze(tf.stack(input_se_2), axis=0)

                input_se_3 = inputs[2]
                input_se_3 = tf.squeeze(tf.stack(input_se_3), axis=0)

                input_se_4 = inputs[3]
                input_se_4 = tf.squeeze(tf.stack(input_se_4), axis=0)

                input_action = inputs[4]
                input_action = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                                                      [(((dim_observation[0] + 3) / 4 + 1) / 2 + 1) / 2,
                                                       (((dim_observation[1] + 3) / 4 + 1) / 2 + 1) / 2])
                conv_3 = tf.concat([input_se_4, input_action], axis=3)
                # input_action = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                #                                       [(((dim_observation[0]+1)/2+1)/2+1)/2, (((dim_observation[1]+1)/2+1)/2+1)/2])
                # full_input = tf.concat([input_action, input_state], axis=3)

                # conv_1 = hrl.utils.Network.conv2ds(input_state,
                #                                shape=[(32, 8, 2)],
                #                                out_flatten=False,
                #                                activation=tf.nn.relu,
                #                                l2=l2,
                #                                var_scope="conv_1")
                #
                # conv_2 = hrl.utils.Network.conv2ds(conv_1,
                #                                    shape=[(32, 3, 2)],
                #                                    out_flatten=False,
                #                                    activation=tf.nn.relu,
                #                                    l2=l2,
                #                                    var_scope="conv_2")
                #
                # conv_3 = hrl.utils.Network.conv2ds(conv_2,
                #                                    shape=[(32, 3, 2)],
                #                                    out_flatten=False,
                #                                    activation=tf.nn.relu,
                #                                    l2=l2,
                #                                    var_scope="conv_3")
                #
                # conv_3 = tf.concat([conv_3, input_action], axis=3)
                #
                conv_4 = hrl.utils.Network.conv2ds(conv_3,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_4")

                # reward
                conv_r1 = hrl.utils.Network.conv2ds(conv_4,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_r1")

                pool_r1 = tf.layers.max_pooling2d(conv_r1, 2, 1)

                pool_r1 = tf.nn.relu(pool_r1)

                pool_r1 = tf.contrib.layers.flatten(pool_r1)

                reward = hrl.utils.Network.layer_fcs(pool_r1, [256], 1,
                                                     activation_hidden=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_state
                concat_0 = tf.concat([input_se_3, conv_4], axis=3)

                conv_5 = hrl.utils.Network.conv2ds(concat_0,
                                                   shape=[(64, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_4_after")


                up_1 = tf.image.resize_images(conv_5, [((dim_observation[0]+3)/4+1)/2, ((dim_observation[1]+3)/4+1)/2])

                concat_1 = tf.concat([input_se_2, up_1], axis=3)

                concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                   shape=[(32, 1, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="concat_1")

                up_2 = tf.image.resize_images(concat_1, [(dim_observation[0]+3)/4, (dim_observation[1]+3)/4])

                concat_2 = tf.concat([input_se_1, up_2], axis=3)

                concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                   shape=[(32, 1, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="concat_2")

                up_3 = tf.image.resize_images(concat_2, [dim_observation[0], dim_observation[1]])

                # concat_3 = tf.concat([input_se_1, up_3], axis=3)

                next_state = hrl.utils.Network.conv2ds(up_3,
                                                     shape=[(3, 1, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="next_state")

                return {"next_state": next_state, "reward": reward}

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

            f_se_1 = create_se_1
            f_se_2 = create_se_2
            f_se_3 = create_se_3
            f_se_4 = create_se_4
            f_ac = create_ac
            f_env = create_env_upsample
            f_rollout = create_rollout
            f_encoder = create_encoder

        super(I2A, self).__init__(env, f_se_1, f_se_2, f_se_3, f_se_4, f_ac, f_env, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)


Experiment.register(I2A, "A3C with I2A for complex observation state experiments")


if __name__ == '__main__':
    Experiment.main()
