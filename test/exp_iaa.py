import sys
sys.path.append(".")
import logging
import numpy as np
import gym
import cv2
import matplotlib.colors as colors
from exp_algorithms import *
from car import *
from hobotrl.tf_dependent.ops import atanh
from hobotrl.environments.environments import *
from hobotrl.playback import Playback, BigPlayback
from hobotrl.network import Utils
from exp_car_flow import F


class MsPacmanI2A(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 episode_n=10000, learning_rate=1e-4, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4), batch_size=32):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = ScaledRewards(env, 0.1)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if (f_tran and f_rollout and f_ac) is None:
            dim_action = env.action_space.n
            dim_observation = env.observation_space.shape

            def create_se(inputs):
                l2 = 1e-7
                input_observation = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_observation,
                                                    shape=[(32, 8, 4), (64, 4, 2), (32, 3, 2)],
                                                    out_flatten=True,
                                                    activation=tf.nn.relu,
                                                    l2=l2,
                                                    var_scope="se_conv")

                se_linear = hrl.utils.Network.layer_fcs(se_conv, [256], 256,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=None,
                                                        l2=l2,
                                                        var_scope="se_linear")
                return {"se": se_linear}

            def create_ac(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                v = hrl.utils.Network.layer_fcs(input_state, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(input_state, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                return {"v": v, "pi": pi}

            def create_rollout(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                # rollout that imitates the A3C policy

                rollout_action = hrl.utils.Network.layer_fcs(input_state, [256], dim_action,
                                                             activation_hidden=tf.nn.relu,
                                                             activation_out=tf.nn.softmax,
                                                             l2=l2,
                                                             var_scope="pi")
                return {"rollout_action": rollout_action}

            def create_transition(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                input_action = inputs[1]
                # input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)

                fc_action = hrl.utils.Network.layer_fcs(input_action, [], 256,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_action")

                concat = tf.multiply(input_state, fc_action)
                # concat = tf.concat([input_state, fc_action], axis=-1)

                fc_out = hrl.utils.Network.layer_fcs(concat, [], 256,
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
                next_state = hrl.utils.Network.layer_fcs(fc_out, [256], 256,
                                                         activation_hidden=tf.nn.relu,
                                                         activation_out=None,
                                                         l2=l2,
                                                         var_scope="next_state")

                return {"next_state": next_state, "reward": reward}

            def create_transition_momentum(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                input_action = inputs[1]
                # input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)

                fc_action = hrl.utils.Network.layer_fcs(input_action, [], 256,
                                                        activation_hidden=tf.nn.relu,
                                                        activation_out=tf.nn.relu,
                                                        l2=l2,
                                                        var_scope="fc_action")

                concat = tf.multiply(input_state, fc_action)

                fc_out = hrl.utils.Network.layer_fcs(concat, [], 256,
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

                # next_goal
                Action_related_goal = hrl.utils.Network.layer_fcs(fc_out, [256], 256,
                                                     activation_hidden=tf.nn.relu,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="TC")

                Action_unrelated_goal = hrl.utils.Network.layer_fcs(input_state, [256], 256,
                                                     activation_hidden=tf.nn.relu,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="TM")
                Action_unrelated_goal = Utils.scale_gradient(Action_unrelated_goal, 1e-3)
                next_goal = Action_related_goal + Action_unrelated_goal

                return {"next_state": next_goal, "reward": reward, "momentum": Action_unrelated_goal,
                        "action_related": Action_related_goal}

            def create_decoder(inputs):
                l2 = 1e-7
                input_goal = inputs[0]

                input_frame = inputs[1]

                conv_1 = hrl.utils.Network.conv2ds(input_frame,
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
                                                   shape=[(32, 3, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_3")

                conv_3_shape = conv_3.shape.as_list()
                fc_1 = hrl.utils.Network.layer_fcs(input_goal, [], conv_3_shape[1] * conv_3_shape[2] *
                                                   (2 * 256 / conv_3_shape[1] / conv_3_shape[2]),
                                                   activation_hidden=tf.nn.relu,
                                                   activation_out=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="fc_goal")

                twoD_out = tf.reshape(fc_1, [-1, conv_3_shape[1], conv_3_shape[2],
                                             2 * 256 / conv_3_shape[1] / conv_3_shape[2]])

                concat_0 = tf.concat([conv_3, twoD_out], axis=3)

                conv_4 = hrl.utils.Network.conv2ds(concat_0,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_5")

                up_1 = tf.image.resize_images(conv_4, conv_2.shape.as_list()[1:3])

                concat_1 = tf.concat([conv_2, up_1], axis=3)
                concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                     shape=[(32, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_1")

                up_2 = tf.image.resize_images(concat_1, conv_1.shape.as_list()[1:3])

                concat_2 = tf.concat([conv_1, up_2], axis=3)
                concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                     shape=[(64, 4, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_2")

                up_3 = tf.image.resize_images(concat_2, input_frame.shape.as_list()[1:3])

                concat_3 = tf.concat([input_frame, up_3], axis=3)
                concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(64, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_3")

                next_frame = hrl.utils.Network.conv2ds(concat_3,
                                                       shape=[(3, 3, 1)],
                                                       out_flatten=False,
                                                       activation=tf.nn.relu,
                                                       l2=l2,
                                                       var_scope="next_frame")

                return {"next_frame": next_frame}

            def create_env_upsample_little(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_state = tf.squeeze(tf.stack(input_state), axis=0)

                input_action = inputs[1]
                input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)
                input_action_tiled = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                                                      [((((dim_observation[0]+1)/2+1)/2+1)/2+1)/2,
                                                       ((((dim_observation[1]+1)/2+1)/2+1)/2+1)/2])

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
                input_argu = inputs[0]
                # input_reward = inputs[1]
                #
                # rse = hrl.utils.Network.conv2ds(input_state,
                #                                 shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                #                                 out_flatten=True,
                #                                 activation=tf.nn.relu,
                #                                 l2=l2,
                #                                 var_scope="rse")
                #
                # re_conv = hrl.utils.Network.layer_fcs(rse, [], 200,
                #                                       activation_hidden=tf.nn.relu,
                #                                       activation_out=tf.nn.relu,
                #                                       l2=l2,
                #                                       var_scope="re_conv")
                #
                # # re_conv = tf.concat([re_conv, tf.reshape(input_reward, [-1, 1])], axis=1)
                # re_conv = tf.concat([re_conv, input_reward], axis=1)

                re = hrl.utils.Network.layer_fcs(input_argu, [256], 256,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="re")

                return {"re": re}

            f_se = create_se
            f_ac = create_ac
            # f_env = create_env_upsample_little
            f_rollout = create_rollout
            f_encoder = create_encoder
            f_tran = create_transition_momentum
            f_decoder = create_decoder

        super(MsPacmanI2A, self).__init__(env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)
Experiment.register(MsPacmanI2A, "A3C with I2A for MsPacman")


class MsPacmanOTDQN(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=16000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=100000,
                 batch_size=16, curriculum=[1, 3, 5], skip_step=[500000, 1000000], sampler_creator=None,
                 asynchronous=False, save_image_interval=10000, with_momentum=True):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = ScaledRewards(env, 0.01)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder()

        super(MsPacmanOTDQN, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum,
                                            skip_step, sampler_creator, asynchronous, save_image_interval,
                                            with_momentum=with_momentum)
Experiment.register(MsPacmanOTDQN, "OTDQN for MsPacman with half input")


class MsPacmanOTDQN_goal_256(MsPacmanOTDQN):
    def __init__(self, with_momentum=False):
        super(MsPacmanOTDQN_goal_256, self).__init__(with_momentum=with_momentum)
Experiment.register(MsPacmanOTDQN_goal_256, "goal OTDQN for MsPacman with half input")


class MsPacmanOTDQN_mom_1600(MsPacmanOTDQN):
    def __init__(self, env=None, episode_n=16000, f_create_q=None, f_se=None, f_transition=None, f_decoder=None,
                 with_momentum=True):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = ScaledRewards(env, 0.01)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if f_se is None:
            f = F(env, 1600)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder()

        super(MsPacmanOTDQN_mom_1600, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder,
                                                     with_momentum=with_momentum)
Experiment.register(MsPacmanOTDQN_mom_1600, "Hidden state size of 1600 on OTDQN for MsPacman with half input")


class MsPacmanOTDQN_goal(MsPacmanOTDQN_mom_1600):
    def __init__(self, with_momentum=False):
        super(MsPacmanOTDQN_goal, self).__init__(with_momentum=with_momentum)
Experiment.register(MsPacmanOTDQN_goal, "goal(1600) OTDQN for MsPacman with half input")


class MsPacmanOTDQN_mom_decoder(MsPacmanOTDQN):
    def __init__(self, env=None, episode_n=16000, f_create_q=None, f_se=None, f_transition=None, f_decoder=None):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = ScaledRewards(env, 0.01)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if f_se is None:
            f = F(env, 256)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder_deconv()

        super(MsPacmanOTDQN_mom_decoder, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder)
Experiment.register(MsPacmanOTDQN_mom_decoder, "Deconv decoder and hidden state size of 1600 on OTDQN for MsPacman with half input")


class OTDQN_ob_MsPacman(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=160000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, curriculum=[1, 3, 5], skip_step=[500000, 1000000], sampler_creator=None,
                 asynchronous=False, save_image_interval=10000, with_ob=True):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = ScaledRewards(env, 0.01)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_env_upsample_fc()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.pass_decoder()
        super(OTDQN_ob_MsPacman, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum, skip_step,
                                            sampler_creator, asynchronous, save_image_interval, with_ob)
Experiment.register(OTDQN_ob_MsPacman, "Old traditional env model with dqn, for MsPacman")


class OTDQN_ob_decoder_MsPacman(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=160000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, curriculum=[1, 3, 5], skip_step=[500000, 1000000], sampler_creator=None,
                 asynchronous=False, save_image_interval=10000, with_ob=True):
        if env is None:
            env = gym.make('MsPacman-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = ScaledRewards(env, 0.01)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_env_deconv_fc()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.pass_decoder()
        super(OTDQN_ob_decoder_MsPacman, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum, skip_step,
                                            sampler_creator, asynchronous, save_image_interval, with_ob)
Experiment.register(OTDQN_ob_decoder_MsPacman, "Old traditional env model with dqn, for MsPacman")


if __name__ == '__main__':
    Experiment.main()
