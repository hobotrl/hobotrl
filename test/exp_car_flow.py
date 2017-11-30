#
# -*- coding: utf-8 -*-


import sys
sys.path.append(".")

from exp_car import *
from hobotrl.tf_dependent.ops import frame_trans, CoordUtil


class I2AFlow(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 episode_n=10000, learning_rate=1e-4, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4), batch_size=16):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)

        if (f_tran and f_rollout and f_ac) is None:
            dim_action = env.action_space.n
            dim_observation = env.observation_space.shape
            chn_se_2d = 32
            dim_se = 256
            nonlinear = tf.nn.elu

            def create_se(inputs):
                l2 = 1e-7
                input_observation = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_observation,

                                               shape=[(8, 8, 4), (16, 4, 2), (chn_se_2d, 3, 2)],
                                               out_flatten=True,
                                               activation=nonlinear,
                                               l2=l2,
                                               var_scope="se_conv")

                se_linear = hrl.utils.Network.layer_fcs(se_conv, [256], dim_se,
                                                        activation_hidden=nonlinear,
                                                        activation_out=None,
                                                        l2=l2,
                                                        var_scope="se_linear")
                return {"se": se_linear}

            def create_ac(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                v = hrl.utils.Network.layer_fcs(input_state, [256, 256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(input_state, [256, 256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                return {"v": v, "pi": pi}

            def create_rollout(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                # rollout that imitates the A3C policy

                rollout_action = hrl.utils.Network.layer_fcs(input_state, [256, 256], dim_action,
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
                                                        activation_hidden=nonlinear,
                                                        activation_out=nonlinear,
                                                        l2=l2,
                                                        var_scope="fc_action")

                concat = tf.multiply(input_state, fc_action)

                fc_out = hrl.utils.Network.layer_fcs(concat, [], 256,
                                                     activation_hidden=nonlinear,
                                                     activation_out=nonlinear,

                                                     l2=l2,
                                                     var_scope="fc_out")

                # reward
                reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                     activation_hidden=nonlinear,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_goal
                Action_related_goal = hrl.utils.Network.layer_fcs(fc_out, [256], dim_se,
                                                     activation_hidden=nonlinear,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="TC")

                Action_unrelated_goal = hrl.utils.Network.layer_fcs(input_state, [256], dim_se,
                                                     activation_hidden=nonlinear,
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

            def create_decoder_deform(inputs):
                l2 = 1e-7
                input_goal = inputs[0]

                input_frame = inputs[1]

                conv_1 = hrl.utils.Network.conv2ds(input_frame,
                                                   shape=[(8, 4, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_1")
                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")
                se0, goal = tf.split(input_goal, 2, axis=1)
                twoD_out = tf.concat((tf.reshape(se0, [-1, 5, 5, dim_se]),
                                      tf.reshape(goal, [-1, 5, 5, dim_se])), axis=-1)
                conv_5 = hrl.utils.Network.conv2ds(twoD_out,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_5")

                up_1 = tf.image.resize_images(conv_5, [((dim_observation[0] + 3) / 4 + 1) / 2,
                                                       ((dim_observation[1] + 3) / 4 + 1) / 2])

                concat_1 = tf.concat([conv_2, up_1], axis=3)
                concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                     shape=[(16, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_1")

                up_2 = tf.image.resize_images(concat_1, [(dim_observation[0] + 3) / 4, (dim_observation[1] + 3) / 4])

                concat_2 = tf.concat([conv_1, up_2], axis=3)
                concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                     shape=[(8, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_2")

                up_3 = tf.image.resize_images(concat_2, [dim_observation[0], dim_observation[1]])

                concat_3 = tf.concat([input_frame, up_3], axis=3)
                concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                     shape=[(8, 3, 1)],
                                                     out_flatten=False,
                                                     activation=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="concat_3")

                next_frame_move = hrl.utils.Network.conv2ds(concat_3,
                                                       shape=[(2, 3, 1)],
                                                       out_flatten=False,
                                                       activation=None,
                                                       l2=l2,
                                                       var_scope="next_frame_move")
                next_frame_move = tf.tanh(next_frame_move) * 16  # within 16 pixels range
                next_frame = frame_trans(input_frame, next_frame_move)
                out = {"next_frame": next_frame}
                return out

            def create_decoder_deform_refine(inputs):
                l2 = 1e-7
                input_goal = inputs[0]
                input_frame = inputs[1]
                frame_shape = tf.shape(input_frame)
                n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
                coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w)
                # normalize to [-1.0, 1.0]
                coord_y = tf.to_float(coord_y) / tf.to_float(h) * 2.0 - 1.0
                coord_x = tf.to_float(coord_x) / tf.to_float(w) * 2.0 - 1.0
                frame_with_coord = tf.concat((input_frame, coord_y, coord_x), axis=3)
                pixel_range = 16.0
                gradient_scale = 16.0
                # /2
                conv_1 = hrl.utils.Network.conv2ds(frame_with_coord,
                                                   shape=[(8, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_1")
                # /4
                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_2")
                # /8
                conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_3")

                goal = hrl.utils.Network.layer_fcs(input_goal, [], 6 * 6 * chn_se_2d,
                                                   activation_hidden=nonlinear,
                                                   activation_out=nonlinear,
                                                   l2=l2, var_scope="fc1")
                twoD_out = tf.reshape(goal, [-1, 6, 6, chn_se_2d])
                conv_se = hrl.utils.Network.conv2ds(twoD_out,
                                                   shape=[(chn_se_2d, 3, 1)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_se")

                up_1 = tf.image.resize_images(conv_se, conv_3.shape.as_list()[1:3])

                concat_1 = tf.concat([conv_3, up_1], axis=3)
                move_8 = hrl.utils.Network.conv2ds(concat_1,
                                                     shape=[(2, 3, 1)],
                                                     out_flatten=False,
                                                     activation=None,
                                                     l2=l2,
                                                     var_scope="move_8")
                move_8 = tf.tanh(move_8 / gradient_scale) * (pixel_range / 8)
                up_2 = hrl.network.Utils.deconv(concat_1, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_2")
                up_move_8 = tf.image.resize_images(move_8, conv_2.shape.as_list()[1:3])
                concat_2 = tf.concat([conv_2, up_2, up_move_8 * 2], axis=3)
                move_4 = hrl.utils.Network.conv2ds(concat_2,
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="move_4")
                move_4 = tf.tanh(move_4 / gradient_scale) * (pixel_range / 4)
                up_3 = hrl.network.Utils.deconv(concat_2, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_3")
                up_move_4 = tf.image.resize_images(move_4, conv_1.shape.as_list()[1:3])
                concat_3 = tf.concat([conv_1, up_3, up_move_4 * 2], axis=3)
                move_2 = hrl.utils.Network.conv2ds(concat_3,
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="move_2")
                move_2 = tf.tanh(move_2 / gradient_scale) * (pixel_range / 2)
                up_4 = hrl.network.Utils.deconv(concat_3, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_4")
                frame_shape = input_frame.shape.as_list()[1:3]
                up_move_2 = tf.image.resize_images(move_2, frame_shape)
                concat_4 = tf.concat([frame_with_coord, up_4, up_move_2 * 2], axis=3)
                move_1 = hrl.utils.Network.conv2ds(concat_4,
                                                       shape=[(2, 3, 1)],
                                                       out_flatten=False,
                                                       activation=None,
                                                       l2=l2,
                                                       var_scope="next_frame_move")
                move_1 = tf.tanh(move_1 / gradient_scale) * pixel_range  # within 16 pixels range
                # weighted sum of different scale
                moves = [(move_1, 16.0),
                         (tf.image.resize_images(move_2, frame_shape) * 2, 4.0),
                         (tf.image.resize_images(move_4, frame_shape) * 4, 1.0),
                         (tf.image.resize_images(move_8, frame_shape) * 8, 0.25)]
                sum_weight = sum([m[1] for m in moves])
                move = tf.add_n([m[0] * m[1] for m in moves]) / sum_weight
                next_frame = frame_trans(input_frame, move, kernel_size=3)
                out = {"next_frame": next_frame}
                return out

            def create_decoder_deform_refine_frames(inputs):
                l2 = 1e-7
                input_goal = inputs[0]
                input_frame = inputs[1]
                frame_shape = tf.shape(input_frame)
                n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
                coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w)
                # normalize to [-1.0, 1.0]
                coord_y = tf.to_float(coord_y) / tf.to_float(h) * 2.0 - 1.0
                coord_x = tf.to_float(coord_x) / tf.to_float(w) * 2.0 - 1.0
                frame_with_coord = tf.concat((input_frame, coord_y, coord_x), axis=3)
                pixel_range = 16.0
                gradient_scale = 16.0
                # /2
                conv_1 = hrl.utils.Network.conv2ds(frame_with_coord,
                                                   shape=[(8, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_1")
                # /4
                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_2")
                # /8
                conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_3")

                goal = hrl.utils.Network.layer_fcs(input_goal, [], 6 * 6 * chn_se_2d,
                                                   activation_hidden=nonlinear,
                                                   activation_out=nonlinear,
                                                   l2=l2, var_scope="fc1")
                twoD_out = tf.reshape(goal, [-1, 6, 6, chn_se_2d])
                conv_se = hrl.utils.Network.conv2ds(twoD_out,
                                                   shape=[(chn_se_2d, 3, 1)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_se")

                up_1 = tf.image.resize_images(conv_se, conv_3.shape.as_list()[1:3])
                frame_8 = tf.image.resize_images(frame_with_coord, conv_3.shape.as_list()[1:3])
                concat_1 = tf.concat([conv_3, up_1, frame_8[:, :, :, 3:5]], axis=3)
                move_8 = hrl.utils.Network.conv2ds(concat_1,
                                                     shape=[(2, 3, 1)],
                                                     out_flatten=False,
                                                     activation=None,
                                                     l2=l2,
                                                     var_scope="move_8")
                move_8 = tf.tanh(move_8 / gradient_scale) * (pixel_range / 8)
                predict_8 = frame_trans(frame_8[:, :, :, 0:3], move_8)
                up_2 = hrl.network.Utils.deconv(concat_1, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_2")
                up_move_8 = tf.image.resize_images(move_8, conv_2.shape.as_list()[1:3]) * 2
                frame_4 = tf.image.resize_images(frame_with_coord, conv_2.shape.as_list()[1:3])
                concat_2 = tf.concat([conv_2, up_2, up_move_8, frame_4[:, :, :, 3:5]], axis=3)
                move_4 = hrl.utils.Network.conv2ds(concat_2,
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="move_4")
                move_4 = tf.tanh(move_4 / gradient_scale) * (pixel_range / 4)
                predict_4 = frame_trans(frame_4[:, :, :, 0:3], move_4)
                up_3 = hrl.network.Utils.deconv(concat_2, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_3")
                up_move_4 = tf.image.resize_images(move_4, conv_1.shape.as_list()[1:3])
                frame_2 = tf.image.resize_images(frame_with_coord, conv_1.shape.as_list()[1:3])
                concat_3 = tf.concat([conv_1, up_3, up_move_4 * 2, frame_2[:, :, :, 3:5]], axis=3)
                move_2 = hrl.utils.Network.conv2ds(concat_3,
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="move_2")
                move_2 = tf.tanh(move_2 / gradient_scale) * (pixel_range / 2)
                predict_2 = frame_trans(frame_2[:, :, :, 0:3], move_2)
                up_4 = hrl.network.Utils.deconv(concat_3, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_4")
                frame_shape = input_frame.shape.as_list()[1:3]
                up_move_2 = tf.image.resize_images(move_2, frame_shape)
                concat_4 = tf.concat([frame_with_coord, up_4, up_move_2 * 2], axis=3)
                move_1 = hrl.utils.Network.conv2ds(concat_4,
                                                       shape=[(2, 3, 1)],
                                                       out_flatten=False,
                                                       activation=None,
                                                       l2=l2,
                                                       var_scope="next_frame_move")
                move_1 = tf.tanh(move_1 / gradient_scale) * pixel_range  # within 16 pixels range
                predict_1 = frame_trans(input_frame, move_1)
                # weighted sum of different scale
                predicts = [
                    (predict_1, 16.0),
                    (tf.image.resize_images(predict_2, frame_shape), 4.0),
                    (tf.image.resize_images(predict_4, frame_shape), 1.0),
                    (tf.image.resize_images(predict_8, frame_shape), 0.25)
                ]
                sum_weight = sum([p[1] for p in predicts])
                next_frame = tf.add_n([p[0] * p[1] for p in predicts]) / sum_weight
                # moves = [(move_1, 16.0),
                #          (tf.image.resize_images(move_2, frame_shape) * 2, 4.0),
                #          (tf.image.resize_images(move_4, frame_shape) * 4, 1.0),
                #          (tf.image.resize_images(move_8, frame_shape) * 8, 0.25)]
                # sum_weight = sum([m[1] for m in moves])
                # move = tf.add_n([m[0] * m[1] for m in moves]) / sum_weight
                # next_frame = frame_trans(input_frame, move, kernel_size=3)
                out = {"next_frame": next_frame}
                return out

            def create_decoder_deform_flow(inputs):
                l2 = 1e-7
                input_goal = inputs[0]
                input_frame = inputs[1]
                frame_shape = tf.shape(input_frame)
                n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
                coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w)
                # normalize to [-1.0, 1.0]
                coord_y = tf.to_float(coord_y) / tf.to_float(h) * 2.0 - 1.0
                coord_x = tf.to_float(coord_x) / tf.to_float(w) * 2.0 - 1.0
                frame_with_coord = tf.concat((input_frame, coord_y, coord_x), axis=3)
                pixel_range = 24.0
                gradient_scale = 16.0
                constrain_flow = False
                # /2
                feature_2 = hrl.utils.Network.conv2ds(frame_with_coord,
                                                   shape=[(8, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_1")
                # /4
                feature_4 = hrl.utils.Network.conv2ds(feature_2,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_2")
                # /8
                feature_8 = hrl.utils.Network.conv2ds(feature_4,
                                                   shape=[(16, 4, 2)],
                                                   out_flatten=False,
                                                   activation=nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_3")

                goal = hrl.utils.Network.layer_fcs(input_goal, [], 6 * 6 * chn_se_2d,
                                                   activation_hidden=nonlinear,
                                                   activation_out=nonlinear,
                                                   l2=l2, var_scope="fc1")
                twoD_out = tf.reshape(goal, [-1, 6, 6, chn_se_2d])
                conv_se = hrl.utils.Network.conv2ds(twoD_out,
                                                    shape=[(chn_se_2d, 3, 1)],
                                                    out_flatten=False,
                                                    activation=nonlinear,
                                                    l2=l2,
                                                    var_scope="conv_se")

                # /8
                trunk_8 = tf.image.resize_images(conv_se, feature_8.shape.as_list()[1:3])
                frame_8 = tf.image.resize_images(frame_with_coord, feature_8.shape.as_list()[1:3])
                trunk_8 = tf.concat([frame_8, feature_8, trunk_8], axis=3)
                flow_8 = hrl.utils.Network.conv2ds(trunk_8,
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="flow_8")
                if constrain_flow:
                    flow_8 = tf.tanh(flow_8 / gradient_scale) * (pixel_range / 8)

                # /4
                trunk_4 = hrl.network.Utils.deconv(trunk_8, kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_4")
                up_flow_8 = tf.image.resize_images(flow_8, feature_4.shape.as_list()[1:3]) * 2
                frame_4 = tf.image.resize_images(frame_with_coord, feature_4.shape.as_list()[1:3])
                trunk_4 = tf.concat([frame_4, feature_4, trunk_4], axis=3)
                trans_4_8 = frame_trans(trunk_4, up_flow_8, name="trans_4_8")
                predict_4_8 = trans_4_8[:, :, :, :3]
                flow_4 = hrl.utils.Network.conv2ds(tf.concat([trans_4_8, up_flow_8], axis=3),
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="flow_4")
                if constrain_flow:
                    flow_4 = tf.tanh(flow_4 / gradient_scale) * (pixel_range / 8)
                flow_sum_4 = flow_4 + up_flow_8

                # /2
                trunk_2 = hrl.network.Utils.deconv(trunk_4[:, :, :, 3:], kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_2")
                up_flow_4 = tf.image.resize_images(flow_sum_4, feature_2.shape.as_list()[1:3]) * 2
                frame_2 = tf.image.resize_images(frame_with_coord, feature_2.shape.as_list()[1:3])
                trunk_2 = tf.concat([frame_2, feature_2, trunk_2], axis=3)
                trans_2_4 = frame_trans(trunk_2, up_flow_4, name="trans_2_4")
                predict_2_4 = trans_2_4[:, :, :, :3]
                flow_2 = hrl.utils.Network.conv2ds(tf.concat([trans_2_4, up_flow_4], axis=3),
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="flow_2")
                if constrain_flow:
                    flow_2 = tf.tanh(flow_2 / gradient_scale) * (pixel_range / 8)
                flow_sum_2 = flow_2 + up_flow_4

                # /1
                frame_shape = input_frame.shape.as_list()[1:3]
                trunk_1 = hrl.network.Utils.deconv(trans_2_4[:, :, :, 3:], kernel_size=3, out_channel=8,
                                                stride=2, activation=nonlinear,
                                                var_scope="up_1")
                up_flow_2 = tf.image.resize_images(flow_sum_2, frame_shape) * 2
                trunk_1 = tf.concat([frame_with_coord, trunk_1], axis=3)
                trans_1_2 = frame_trans(trunk_1, up_flow_2)
                predict_1_2 = trans_1_2[:, :, :, :3]
                flow_1 = hrl.utils.Network.conv2ds(tf.concat([trans_1_2, up_flow_2], axis=3),
                                                   shape=[(2, 3, 1)],
                                                   out_flatten=False,
                                                   activation=None,
                                                   l2=l2,
                                                   var_scope="flow_1")
                if constrain_flow:
                    flow_1 = tf.tanh(flow_1 / gradient_scale) * (pixel_range / 8)
                flow_sum_1 = flow_1 + up_flow_2
                predict_1 = frame_trans(input_frame, flow_sum_1)

                # weighted sum of different scale
                predicts = [
                    (predict_1, 16.0),
                    (predict_1_2, 4.0),
                    (tf.image.resize_images(predict_2_4, frame_shape), 1.0),
                    (tf.image.resize_images(predict_4_8, frame_shape), 0.25)
                ]
                # sum_weight = sum([p[1] for p in predicts])
                # next_frame = tf.add_n([p[0] * p[1] for p in predicts]) / sum_weight
                out = {"next_frame": predict_1,
                       "frame_2": predict_1_2,
                       "frame_4": predict_2_4,
                       "frame_8": predict_4_8}
                return out

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
            # f_decoder = create_decoder
            f_decoder = create_decoder_deform_flow

        super(I2AFlow, self).__init__(env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)
Experiment.register(I2AFlow, "A3C with I2A for CarRacing")


if __name__ == '__main__':
    Experiment.main()