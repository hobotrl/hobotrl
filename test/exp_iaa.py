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


class I2A(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_env=None, f_rollout=None, f_encoder = None, episode_n=10000,
                 learning_rate=1e-4, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = envs.DownsampledMsPacman(env)
            env = envs.ScaledRewards(env, 0.1)
            env = envs.MaxAndSkipEnv(env, skip=4, max_len=4)
            env = envs.FrameStack(env, k=4)
            # env = wrap_car(env, 3, 3, frame=4)

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
                                               var_scope="se")
                return {"se": se_conv}

            def create_ac(inputs):
                l2 = 1e-7
                input_feature = inputs[0]
                #
                # ac_feature = hrl.utils.Network.conv2ds(input_feature,
                #                                    shape=[(32, 3, 1)],
                #                                    out_flatten=True,
                #                                    activation=tf.nn.relu,
                #                                    l2=l2,
                #                                    var_scope="conv_ac")
                se_linear = hrl.utils.Network.layer_fcs(input_feature, [256], 256,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="se_linear")

                v = hrl.utils.Network.layer_fcs(se_linear, [], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(se_linear, [], dim_action,
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

            def create_env_upsample_fc(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_state = tf.squeeze(tf.stack(input_state), axis=0)

                input_action = inputs[1]
                input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)
                # input_action_tiled = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                #                                       [((((dim_observation[0]+1)/2+1)/2+1)/2+1)/2,
                #                                        ((((dim_observation[1]+1)/2+1)/2+1)/2+1)/2])

                conv_1 = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="conv_1")

                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(64, 4, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")

                conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                                   shape=[(64, 3, 2)],
                                                   out_flatten=True,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_3")

                fc_1 = hrl.utils.Network.layer_fcs(conv_3, [], 64*5*5,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_1")

                # concat_action = tf.concat([conv_4, input_action_tiled], axis=3)
                fc_action = hrl.utils.Network.layer_fcs(tf.to_float(input_action), [], 64*5*5,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_action")

                concat = tf.multiply(fc_1, fc_action)

                fc_out = hrl.utils.Network.layer_fcs(concat, [64*5*5], 64*5*5,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_out")

                # reward
                reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                     activation_hidden=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_state
                twoD_out = tf.reshape(fc_out, [-1, 64, 5, 5])

                conv_5 = hrl.utils.Network.conv2ds(twoD_out,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_5")

                up_1 = tf.image.resize_images(conv_5, [((dim_observation[0]+3)/4+1)/2, ((dim_observation[1]+3)/4+1)/2])

                concat_1 = tf.concat([conv_2, up_1], axis=3)
                concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="concat_1")

                up_2 = tf.image.resize_images(concat_1, [(dim_observation[0]+3)/4, (dim_observation[1]+3)/4])

                concat_2 = tf.concat([conv_1, up_2], axis=3)
                concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                   shape=[(64, 4, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="concat_2")

                up_3 = tf.image.resize_images(concat_2, [dim_observation[0], dim_observation[1]])

                concat_3 = tf.concat([input_state, up_3], axis=3)
                concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(64, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_3")

                next_state = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(3, 3, 1)],
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

            f_se = create_se
            f_ac = create_ac
            f_env = create_env_upsample_fc
            f_rollout = create_rollout
            f_encoder = create_encoder

        super(I2A, self).__init__(env, f_se, f_ac, f_env, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)


Experiment.register(I2A, "A3C with I2A for complex observation state experiments")


if __name__ == '__main__':
    Experiment.main()
