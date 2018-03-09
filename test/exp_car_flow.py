#
# -*- coding: utf-8 -*-


import sys
import logging

sys.path.append(".")

import tensorflow.contrib.slim as slim
from exp_car import *
from hobotrl.tf_dependent.ops import frame_trans, CoordUtil
from playground.initialD.ros_environments.clients import DrSimDecisionK8STopView
from hobotrl.environments.environments import RemapFrame
from playground.ot_model import OTModel
from hobotrl.async import AsynchronousAgent
from hobotrl.utils import CappedLinear


class F(object):
    def __init__(self, env, dim_se=256):
        super(F, self).__init__()
        self._env = env
        self.dim_action = self._env.action_space.n
        self.dim_observation = self._env.observation_space.shape
        self.chn_se_2d = 32
        self.dim_se = dim_se
        self.nonlinear = tf.nn.elu
        
    def create_se(self):
        def create_se(inputs):
            l2 = 1e-7
            input_observation = inputs[0]
            se_conv = hrl.utils.Network.conv2ds(input_observation,
        
                                                shape=[(8, 8, 4), (16, 4, 2), (self.chn_se_2d, 3, 2)],
                                                out_flatten=True,
                                                activation=self.nonlinear,
                                                l2=l2,
                                                var_scope="se_conv")
        
            se_linear = hrl.utils.Network.layer_fcs(se_conv, [self.dim_se], self.dim_se,
                                                    activation_hidden=self.nonlinear,
                                                    activation_out=None,
                                                    l2=l2,
                                                    var_scope="se_linear")
            return {"se": se_linear}
        return create_se

    def create_se_channels(self):
        def create_se(inputs):
            l2 = 1e-7
            input_observation = inputs[0]
            se_conv = hrl.utils.Network.conv2ds(input_observation,

                                                shape=[(32, 8, 4), (64, 4, 2), (self.chn_se_2d, 3, 2)],
                                                out_flatten=True,
                                                activation=self.nonlinear,
                                                l2=l2,
                                                var_scope="se_conv")

            se_linear = hrl.utils.Network.layer_fcs(se_conv, [self.dim_se], self.dim_se,
                                                    activation_hidden=self.nonlinear,
                                                    activation_out=None,
                                                    l2=l2,
                                                    var_scope="se_linear")
            return {"se": se_linear}

        return create_se

    def create_ac(self):
        def create_ac(inputs):
            l2 = 1e-7
            input_state = inputs[0]
        
            v = hrl.utils.Network.layer_fcs(input_state, [256, 256], 1,
                                            activation_hidden=self.nonlinear,
                                            l2=l2,
                                            var_scope="v")
            v = tf.squeeze(v, axis=1)
            pi = hrl.utils.Network.layer_fcs(input_state, [256, 256], self.dim_action,
                                             activation_hidden=self.nonlinear,
                                             activation_out=tf.nn.softmax,
                                             l2=l2,
                                             var_scope="pi")
        
            return {"v": v, "pi": pi}
        return create_ac

    def create_q(self):
        def create_q(inputs):
            l2 = 1e-7
            input_state = inputs[0]

            q = hrl.utils.Network.layer_fcs(input_state, [256, 256], self.dim_action,
                                            activation_hidden=self.nonlinear,
                                            l2=l2,
                                            var_scope="q")
            return {"q": q}

        return create_q

    def create_rollout(self):
        def create_rollout(inputs):
            l2 = 1e-7
            input_state = inputs[0]
        
            # rollout that imitates the A3C policy
        
            rollout_action = hrl.utils.Network.layer_fcs(input_state, [256, 256], self.dim_action,
                                                         activation_hidden=self.nonlinear,
                                                         activation_out=tf.nn.softmax,
                                                         l2=l2,
                                                         var_scope="pi")
            return {"rollout_action": rollout_action}
        return create_rollout

    def create_transition(self):
        def create_transition(inputs):
            l2 = 1e-7
            input_state = inputs[0]
        
            input_action = inputs[1]
            # input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)
        
            fc_action = hrl.utils.Network.layer_fcs(input_action, [], self.dim_se,
                                                    activation_hidden=self.nonlinear,
                                                    activation_out=self.nonlinear,
                                                    l2=l2,
                                                    var_scope="fc_action")
        
            concat = tf.multiply(input_state, fc_action)
            # concat = tf.concat([input_state, fc_action], axis=-1)
        
            fc_out = hrl.utils.Network.layer_fcs(concat, [], self.dim_se,
                                                 activation_hidden=self.nonlinear,
                                                 activation_out=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="fc_out")
        
            # reward
            reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                 activation_hidden=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="reward")
            reward = tf.squeeze(reward, axis=1)
        
            # next_state
            next_state = hrl.utils.Network.layer_fcs(fc_out, [self.dim_se], self.dim_se,
                                                     activation_hidden=self.nonlinear,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="next_state")
        
            return {"next_state": next_state, "reward": reward}
        return create_transition

    def create_transition_momentum(self):
        def create_transition_momentum(inputs):
            l2 = 1e-7
            input_state = inputs[0]
        
            input_action = inputs[1]
            # input_action = tf.one_hot(indices=input_action, depth=dim_action, on_value=1.0, off_value=0.0, axis=-1)
        
            fc_action = hrl.utils.Network.layer_fcs(input_action, [], self.dim_se,
                                                    activation_hidden=self.nonlinear,
                                                    activation_out=self.nonlinear,
                                                    l2=l2,
                                                    var_scope="fc_action")
        
            concat = tf.multiply(input_state, fc_action)
        
            fc_out = hrl.utils.Network.layer_fcs(concat, [], self.dim_se,
                                                 activation_hidden=self.nonlinear,
                                                 activation_out=self.nonlinear,
        
                                                 l2=l2,
                                                 var_scope="fc_out")
        
            # reward
            reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                 activation_hidden=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="reward")
            reward = tf.squeeze(reward, axis=1)
        
            # next_goal
            Action_related_goal = hrl.utils.Network.layer_fcs(fc_out, [self.dim_se], self.dim_se,
                                                              activation_hidden=self.nonlinear,
                                                              activation_out=None,
                                                              l2=l2,
                                                              var_scope="TC")
        
            Action_unrelated_goal = hrl.utils.Network.layer_fcs(input_state, [self.dim_se], self.dim_se,
                                                                activation_hidden=self.nonlinear,
                                                                activation_out=None,
                                                                l2=l2,
                                                                var_scope="TM")
            Action_unrelated_goal = Utils.scale_gradient(Action_unrelated_goal, 1e-3)
            next_goal = Action_related_goal + Action_unrelated_goal
        
            return {"next_state": next_goal, "reward": reward, "momentum": Action_unrelated_goal,
                    "action_related": Action_related_goal}
        return create_transition_momentum
        
    def create_decoder(self):
        def create_decoder(inputs):
            l2 = 1e-7
            input_goal = inputs[0]
        
            input_frame = inputs[1]
        
            conv_1 = hrl.utils.Network.conv2ds(input_frame,
                                               shape=[(32, 8, 4)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_1")
            conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                               shape=[(64, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_2")
        
            conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                               shape=[(32, 3, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_3")
        
            conv_3_shape = conv_3.shape.as_list()
            fc_1 = hrl.utils.Network.layer_fcs(input_goal, [], conv_3_shape[1] * conv_3_shape[2] *
                                               (2 * self.dim_se / conv_3_shape[1] / conv_3_shape[2]),
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2,
                                               var_scope="fc_goal")
        
            twoD_out = tf.reshape(fc_1, [-1, conv_3_shape[1], conv_3_shape[2],
                                         2 * self.dim_se / conv_3_shape[1] / conv_3_shape[2]])
        
            concat_0 = tf.concat([conv_3, twoD_out], axis=3)
        
            conv_4 = hrl.utils.Network.conv2ds(concat_0,
                                               shape=[(32, 3, 1)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_5")
        
            up_1 = tf.image.resize_images(conv_4, conv_2.shape.as_list()[1:3])
        
            concat_1 = tf.concat([conv_2, up_1], axis=3)
            concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                 shape=[(32, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_1")
        
            up_2 = tf.image.resize_images(concat_1, conv_1.shape.as_list()[1:3])
        
            concat_2 = tf.concat([conv_1, up_2], axis=3)
            concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                 shape=[(64, 4, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_2")
        
            up_3 = tf.image.resize_images(concat_2, input_frame.shape.as_list()[1:3])
        
            concat_3 = tf.concat([input_frame, up_3], axis=3)
            concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                 shape=[(64, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_3")
        
            next_frame = hrl.utils.Network.conv2ds(concat_3,
                                                   shape=[(3, 3, 1)],
                                                   out_flatten=False,
                                                   activation=self.nonlinear,
                                                   l2=l2,
                                                   var_scope="next_frame")
        
            return {"next_frame": next_frame}
        return create_decoder

    def create_decoder_channel(self):
        def create_decoder(inputs):
            l2 = 1e-7
            input_goal = inputs[0]

            input_frame = inputs[1]

            conv_1 = hrl.utils.Network.conv2ds(input_frame,
                                               shape=[(8, 8, 4)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_1")
            conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                               shape=[(16, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_2")

            conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                               shape=[(self.chn_se_2d, 3, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_3")

            conv_3_shape = conv_3.shape.as_list()
            fc_1 = hrl.utils.Network.layer_fcs(input_goal, [], conv_3_shape[1] * conv_3_shape[2] *
                                               (2 * self.dim_se / conv_3_shape[1] / conv_3_shape[2]),
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2,
                                               var_scope="fc_goal")

            twoD_out = tf.reshape(fc_1, [-1, conv_3_shape[1], conv_3_shape[2],
                                         2 * self.dim_se / conv_3_shape[1] / conv_3_shape[2]])

            concat_0 = tf.concat([conv_3, twoD_out], axis=3)

            conv_4 = hrl.utils.Network.conv2ds(concat_0,
                                               shape=[(self.chn_se_2d, 3, 1)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_5")

            up_1 = tf.image.resize_images(conv_4, conv_2.shape.as_list()[1:3])

            concat_1 = tf.concat([conv_2, up_1], axis=3)
            concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                 shape=[(self.chn_se_2d, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_1")

            up_2 = tf.image.resize_images(concat_1, conv_1.shape.as_list()[1:3])

            concat_2 = tf.concat([conv_1, up_2], axis=3)
            concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                 shape=[(16, 4, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_2")

            up_3 = tf.image.resize_images(concat_2, input_frame.shape.as_list()[1:3])

            concat_3 = tf.concat([input_frame, up_3], axis=3)
            concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                 shape=[(8, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_3")

            next_frame = hrl.utils.Network.conv2ds(concat_3,
                                                   shape=[(3, 3, 1)],
                                                   out_flatten=False,
                                                   activation=self.nonlinear,
                                                   l2=l2,
                                                   var_scope="next_frame")

            return {"next_frame": next_frame}

        return create_decoder

    def create_decoder_deconv(self):
        def create_decoder_deconv(inputs):
            l2 = 1e-7
            input_goal = inputs[0]
            input_frame = inputs[1]

            height = ((((self.dim_observation[0] + 3) / 4 + 1) / 2) + 1) / 2
            width = ((((self.dim_observation[1] + 3) / 4 + 1) / 2) + 1) / 2

            resized_goal = hrl.utils.Network.layer_fcs(input_goal, [], height * width,
                                                       activation_hidden=self.nonlinear,
                                                       activation_out=self.nonlinear,
                                                       l2=l2,
                                                       var_scope="resized_goal")

            twoDgoal = tf.reshape(resized_goal, [-1, height, width, 1])

            next_frame_before = hrl.utils.Network.conv2ds_transpose(twoDgoal,
                                                                    shape=[(self.chn_se_2d, 3, 2), (16, 4, 2), (8, 8, 4)],
                                                                    activation=self.nonlinear,
                                                                    l2=l2,
                                                                    var_scope="next_frame_before")

            next_frame = hrl.utils.Network.conv2ds(next_frame_before,
                                                   shape=[(3, 3, 1)],
                                                   out_flatten=False,
                                                   activation=self.nonlinear,
                                                   l2=l2,
                                                   var_scope="next_frame")

            return {"next_frame": next_frame}

        return create_decoder_deconv

    def create_decoder_deform(self):
        def create_decoder_deform(inputs):
            l2 = 1e-7
            input_goal = inputs[0]
        
            input_frame = inputs[1]
        
            conv_1 = hrl.utils.Network.conv2ds(input_frame,
                                               shape=[(8, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_1")
            conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                               shape=[(16, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_2")
            se0, goal = tf.split(input_goal, 2, axis=1)
            twoD_out = tf.concat((tf.reshape(se0, [-1, 5, 5, self.dim_se]),
                                  tf.reshape(goal, [-1, 5, 5, self.dim_se])), axis=-1)
            conv_5 = hrl.utils.Network.conv2ds(twoD_out,
                                               shape=[(32, 3, 1)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_5")
        
            up_1 = tf.image.resize_images(conv_5, [((self.dim_observation[0] + 3) / 4 + 1) / 2,
                                                   ((self.dim_observation[1] + 3) / 4 + 1) / 2])
        
            concat_1 = tf.concat([conv_2, up_1], axis=3)
            concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                 shape=[(16, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_1")
        
            up_2 = tf.image.resize_images(concat_1, [(self.dim_observation[0] + 3) / 4, (self.dim_observation[1] + 3) / 4])
        
            concat_2 = tf.concat([conv_1, up_2], axis=3)
            concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                 shape=[(8, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_2")
        
            up_3 = tf.image.resize_images(concat_2, [self.dim_observation[0], self.dim_observation[1]])
        
            concat_3 = tf.concat([input_frame, up_3], axis=3)
            concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                 shape=[(8, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
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
        return create_decoder_deform
        
    def create_decoder_deform_refine(self):
        def create_decoder_deform_refine(inputs):

            l2 = 1e-7
            input_goal = inputs[0]
            input_frame = inputs[1]
            frame_shape = tf.shape(input_frame)
            n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
            coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w, normalize=True)
            frame_with_coord = tf.concat((input_frame, coord_y, coord_x), axis=3)
            pixel_range = 16.0
            gradient_scale = 16.0
            # /2
            conv_1 = hrl.utils.Network.conv2ds(frame_with_coord,
                                               shape=[(8, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_1")
            # /4
            conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                               shape=[(16, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_2")
            # /8
            conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                               shape=[(16, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_3")
        
            goal = hrl.utils.Network.layer_fcs(input_goal, [], 6 * 6 * self.chn_se_2d,
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2, var_scope="fc1")
            twoD_out = tf.reshape(goal, [-1, 6, 6, self.chn_se_2d])
            conv_se = hrl.utils.Network.conv2ds(twoD_out,
                                                shape=[(self.chn_se_2d, 3, 1)],
                                                out_flatten=False,
                                                activation=self.nonlinear,
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
                                            stride=2, activation=self.nonlinear,
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
                                            stride=2, activation=self.nonlinear,
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
                                            stride=2, activation=self.nonlinear,
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
        return create_decoder_deform_refine
        
    def create_decoder_deform_refine_frames(self):
        def create_decoder_deform_refine_frames(inputs):
            l2 = 1e-7
            input_goal = inputs[0]
            input_frame = inputs[1]
            frame_shape = tf.shape(input_frame)
            n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
            coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w, normalize=True)
            frame_with_coord = tf.concat((input_frame, coord_y, coord_x), axis=3)
            pixel_range = 16.0
            gradient_scale = 16.0
            # /2
            conv_1 = hrl.utils.Network.conv2ds(frame_with_coord,
                                               shape=[(8, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_1")
            # /4
            conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                               shape=[(16, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_2")
            # /8
            conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                               shape=[(16, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_3")
        
            goal = hrl.utils.Network.layer_fcs(input_goal, [], 6 * 6 * self.chn_se_2d,
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2, var_scope="fc1")
            twoD_out = tf.reshape(goal, [-1, 6, 6, self.chn_se_2d])
            conv_se = hrl.utils.Network.conv2ds(twoD_out,
                                                shape=[(self.chn_se_2d, 3, 1)],
                                                out_flatten=False,
                                                activation=self.nonlinear,
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
                                            stride=2, activation=self.nonlinear,
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
                                            stride=2, activation=self.nonlinear,
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
                                            stride=2, activation=self.nonlinear,
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
        return create_decoder_deform_refine_frames
        
    def create_decoder_deform_flow(self):
        def create_decoder_deform_flow(inputs):
            l2 = 1e-7
            input_goal = inputs[0]
            input_frame = inputs[1]
            frame_shape = tf.shape(input_frame)
            n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
            coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w, normalize=True)
            frame_with_coord = tf.concat((input_frame, coord_y, coord_x), axis=3)
            pixel_range = 24.0
            gradient_scale = 16.0
            constrain_flow = False
            flow_residual = True
            # /2
            feature_2 = hrl.utils.Network.conv2ds(frame_with_coord,
                                                  shape=[(8, 4, 2)],
                                                  out_flatten=False,
                                                  activation=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="conv_1")
            # /4
            feature_4 = hrl.utils.Network.conv2ds(feature_2,
                                                  shape=[(16, 4, 2)],
                                                  out_flatten=False,
                                                  activation=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="conv_2")
            # /8
            feature_8 = hrl.utils.Network.conv2ds(feature_4,
                                                  shape=[(32, 4, 2)],
                                                  out_flatten=False,
                                                  activation=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="conv_3")
            # /16
            feature_16 = hrl.utils.Network.conv2ds(feature_8,
                                                   shape=[(64, 4, 2)],
                                                   out_flatten=False,
                                                   activation=self.nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_4")
            feature_16_hw = feature_16.shape.as_list()[1:3]
            goal = hrl.utils.Network.layer_fcs(input_goal, [], feature_16_hw[0] * feature_16_hw[1] * self.chn_se_2d,
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2, var_scope="fc1")
            twoD_out = tf.reshape(goal, [-1, feature_16_hw[0], feature_16_hw[1], self.chn_se_2d])
            frame_16 = tf.image.resize_images(frame_with_coord, feature_16_hw)
            trunk_16 = tf.concat([frame_16, feature_16, twoD_out], axis=3)
            flow_16 = hrl.utils.Network.conv2ds(trunk_16,
                                                shape=[(2, 3, 1)],
                                                out_flatten=False,
                                                activation=None,
                                                l2=l2,
                                                var_scope="flow_16")
            if constrain_flow:
                flow_16 = tf.tanh(flow_16 / gradient_scale) * (pixel_range / 8)
        
            # /8
            trunk_8 = hrl.network.Utils.deconv(trunk_16, kernel_size=3, out_channel=32,
                                               stride=2, activation=self.nonlinear,
                                               var_scope="up_8")
            up_flow_16 = tf.image.resize_images(flow_16, feature_8.shape.as_list()[1:3]) * 2
            frame_8 = tf.image.resize_images(frame_with_coord, feature_8.shape.as_list()[1:3])
            trunk_8 = tf.concat([frame_8, feature_8, trunk_8], axis=3)
            trans_8_16 = frame_trans(trunk_8, up_flow_16, name="trans_8_16")
            predict_8_16 = trans_8_16[:, :, :, :3]
            flow_8 = hrl.utils.Network.conv2ds(tf.concat([trans_8_16, up_flow_16], axis=3),
                                               shape=[(2, 3, 1)],
                                               out_flatten=False,
                                               activation=None,
                                               l2=l2,
                                               var_scope="flow_8")
            if constrain_flow:
                flow_8 = tf.tanh(flow_8 / gradient_scale) * (pixel_range / 8)
            if flow_residual:
                flow_sum_8 = flow_8 + up_flow_16
            else:
                flow_sum_8 = flow_8
            # /4
            trunk_4 = hrl.network.Utils.deconv(trunk_8, kernel_size=3, out_channel=16,
                                               stride=2, activation=self.nonlinear,
                                               var_scope="up_4")
            up_flow_8 = tf.image.resize_images(flow_sum_8, feature_4.shape.as_list()[1:3]) * 2
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
            if flow_residual:
                flow_sum_4 = flow_4 + up_flow_8
            else:
                flow_sum_4 = flow_4
            # /2
            trunk_2 = hrl.network.Utils.deconv(trunk_4[:, :, :, 3:], kernel_size=3, out_channel=8,
                                               stride=2, activation=self.nonlinear,
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
            if flow_residual:
                flow_sum_2 = flow_2 + up_flow_4
            else:
                flow_sum_2 = flow_2
            # /1
            frame_shape = input_frame.shape.as_list()[1:3]
            trunk_1 = hrl.network.Utils.deconv(trans_2_4[:, :, :, 3:], kernel_size=3, out_channel=8,
                                               stride=2, activation=self.nonlinear,
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
            if flow_residual:
                flow_sum_1 = flow_1 + up_flow_2
            else:
                flow_sum_1 = flow_1
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
                   "frame_8": predict_4_8,
                   "frame_16": predict_8_16,
                   "flow": flow_sum_1}
            return out
        return create_decoder_deform_flow

    def decoder_multiflow(self):
        feature_channel_n = 2
        l2 = 1e-5

        def feature_extractor(input_image):
            # /1
            feature_1 = hrl.utils.Network.conv2ds(input_image,
                                                  shape=[(8, 3, 1)],
                                                  out_flatten=False,
                                                  activation=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="conv_0")
            # /2
            feature_2 = hrl.utils.Network.conv2ds(feature_1,
                                                  shape=[(8, 4, 2)],
                                                  out_flatten=False,
                                                  activation=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="conv_1")
            # /4
            feature_4 = hrl.utils.Network.conv2ds(feature_2,
                                                  shape=[(16, 4, 2)],
                                                  out_flatten=False,
                                                  activation=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="conv_2")
            # /8
            feature_8 = hrl.utils.Network.conv2ds(feature_4,
                                                  shape=[(32, 4, 2)],
                                                  out_flatten=False,
                                                  activation=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="conv_3")
            # /16
            feature_16 = hrl.utils.Network.conv2ds(feature_8,
                                                   shape=[(64, 4, 2)],
                                                   out_flatten=False,
                                                   activation=self.nonlinear,
                                                   l2=l2,
                                                   var_scope="conv_4")
            return {"features": [feature_1, feature_2, feature_4, feature_8, feature_16]}

        def fcn(feature_tower, image):
            out = None
            reverse_feature_tower = []
            for i in range(len(feature_tower)):
                feature = feature_tower[len(feature_tower)-1-i]
                feature_shape = feature.shape.as_list()
                if out is not None:
                    out = tf.image.resize_images(out, feature_shape[1:3])
                    out = tf.concat([out, feature], axis=3)
                    out = hrl.utils.Network.conv2d(out, 1, 1, feature_shape[3], (1, 1),
                                                   activation=self.nonlinear,
                                                   l2=l2, var_scope="fcn%d" % i)
                else:
                    out = feature
                reverse_feature_tower.append(out)
            out = tf.image.resize_images(out, image.shape.as_list()[1:3])
            segment_class = hrl.utils.Network.conv2d(out, 1, 1, feature_channel_n, (1, 1),
                                                   activation=self.nonlinear,
                                                   l2=l2, var_scope="segment")
            segment_class = tf.nn.softmax(segment_class, dim=3)
            return {"feature": reverse_feature_tower,
                    "class": segment_class}

        def pwc_resflow_multi_simple(input_goal, reverse_feature_tower, input_image, separate_kernel=False):
            # transport
            frame_shape = tf.shape(input_image)
            n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
            coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w, normalize=True)
            ones = tf.ones(shape=(n, h, w, 1), dtype=tf.float32)
            # homogeneous coordinates: (y, x, 1)
            coord = tf.concat((coord_y, coord_x, ones), axis=3)
            shape_short = reverse_feature_tower[0].shape.as_list()[1:3]
            kernel_h, kernel_w = 1, 1
            goal = hrl.utils.Network.layer_fcs(input_goal, [], 256,
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2, var_scope="fc1")
            flows, weights = [], []
            for i in range(len(reverse_feature_tower)):
                feature = reverse_feature_tower[i]
                h, w = feature.shape.as_list()[1:3]
                sub_coord = tf.image.resize_images(coord, (h, w))
                feature_to_warp = sub_coord
                if len(flows) == 0:
                    up_flow = None
                else:
                    up_flow = tf.image.resize_images(flows[-1], (h, w)) * 2
                    up_weight = tf.image.resize_images(weights[-1], (h, w))
                    logging.warning("flow shape:%s", up_flow)
                    feature_to_warp = warp_trunk_feature(
                        [],
                        sub_coord,
                        up_flow)
                    if not separate_kernel:
                        feature_to_warp = tf.concat(feature_to_warp + [sub_coord], axis=3)
                if not separate_kernel:
                    channel_in = feature_to_warp.shape.as_list()[3]
                    channel_out = 3 * feature_channel_n
                    # channel_in * h * w * ( (flow + weight) * channel )
                    kernel_size = channel_in * kernel_h * kernel_w * channel_out
                    conv_kernel = hrl.network.Utils.layer_fcs(goal, [], kernel_size,
                                                              activation_hidden=self.nonlinear,
                                                              activation_out=self.nonlinear,
                                                              l2=l2, var_scope="kernel_%d" % i)
                    conv_kernel = tf.reshape(conv_kernel, (-1, channel_in, channel_out))
                    # conv_bias = tf.get_variable('conv_bias_%d' % i, [1, 1, 1, channel_out],
                    #                             initializer=tf.constant_initializer(0),
                    #                             trainable=True)
                    feature_matrix = tf.reshape(feature_to_warp, (-1, h*w, channel_in))
                    feature_matrix = tf.matmul(feature_matrix, conv_kernel)
                    feature_matrix = tf.reshape(feature_matrix, (-1, h, w, channel_out))
                    flow_weight = feature_matrix
                    logging.warning("flow_weight shape:%s", flow_weight)
                    flow, weight = flow_weight[:, :, :, 0:2 * feature_channel_n], \
                                   flow_weight[:, :, :, 2 * feature_channel_n:]
                else:
                    # separate channel
                    if type(feature_to_warp) != list:
                        # first level, coord only
                        feature_to_warp = [feature_to_warp] * feature_channel_n
                    channeled_flow, channeled_weight = [], []
                    for j in range(feature_channel_n):
                        warped_feature = feature_to_warp[j]
                        warped_feature = tf.concat([warped_feature, sub_coord], axis=3)
                        channel_in = warped_feature.shape.as_list()[3]
                        channel_out = 3
                        kernel_size = channel_in * kernel_h * kernel_w * channel_out
                        conv_kernel = hrl.network.Utils.layer_fcs(goal, [], kernel_size,
                                                                  activation_hidden=self.nonlinear,
                                                                  activation_out=self.nonlinear,
                                                                  l2=l2, var_scope="kernel_%d_%d" % (i, j))
                        conv_kernel = tf.reshape(conv_kernel, (-1, channel_in, channel_out))
                        feature_matrix = tf.reshape(warped_feature, (-1, h*w, channel_in))
                        feature_matrix = tf.matmul(feature_matrix, conv_kernel)
                        feature_matrix = tf.reshape(feature_matrix, (-1, h, w, channel_out))
                        flow_weight = feature_matrix
                        logging.warning("flow_weight shape:%s", flow_weight)
                        flow, weight = flow_weight[:, :, :, 0:2], \
                                       flow_weight[:, :, :, 2:]
                        channeled_flow.append(flow)
                        channeled_weight.append(weight)
                    flow = tf.concat(channeled_flow, axis=3)
                    weight = tf.concat(channeled_weight, axis=3)

                logging.warning("flow:%s, weight :%s", flow, weight)
                if up_flow is not None:
                    flow = flow + up_flow
                weight = tf.nn.sigmoid(weight) * 1.2
                flows.append(flow)
                weights.append(weight)
            return {"motion": [m for m in reversed(flows)], "weight": [w for w in reversed(weights)]}

        def pwc_resflow_multi(input_goal, reverse_feature_tower, input_image):
            # predict optical flows and weights for each image channels
            frame_shape = tf.shape(input_image)
            n, h, w, c = frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]
            coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w, normalize=True)
            coord = tf.concat((coord_y, coord_x), axis=3)
            shape_short = reverse_feature_tower[0].shape.as_list()[1:3]
            goal = hrl.utils.Network.layer_fcs(input_goal, [], shape_short[0] * shape_short[1] * self.chn_se_2d,
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2, var_scope="fc1")

            twoD_out = tf.reshape(goal, [-1, shape_short[0], shape_short[1], self.chn_se_2d])
            trunk = twoD_out
            flows, weights = [], []
            for i in range(len(reverse_feature_tower)):
                feature = reverse_feature_tower[i]
                h, w = feature.shape.as_list()[1:3]
                sub_coord = tf.image.resize_images(coord, (h, w))
                feature_to_warp = feature
                if len(flows) == 0:
                    up_flow = None
                else:
                    up_flow = tf.image.resize_images(flows[-1], (h, w)) * 2
                    up_weight = tf.image.resize_images(weights[-1], (h, w))
                    logging.warning("flow shape:%s", up_flow)
                    feature_to_warp = warp_trunk_feature(
                        [feature_to_warp],
                        tf.concat([trunk, sub_coord], axis=3),
                        up_flow)
                    feature_to_warp = tf.concat(feature_to_warp + [up_weight, up_flow], axis=3)
                feature_to_warp = tf.concat([feature_to_warp, sub_coord], axis=3)
                flow_weight = hrl.network.Utils.conv2ds(feature_to_warp,
                                          shape=[(3 * feature_channel_n, 3, 1)],
                                          out_flatten=False,
                                          activation=None,
                                          l2=l2,
                                          var_scope="flow_weight_%d" % i)
                logging.warning("flow_weight shape:%s", flow_weight)
                flow, weight = flow_weight[:, :, :, 0:2 * feature_channel_n], flow_weight[:, :, :, 2*feature_channel_n:]
                logging.warning("flow:%s, weight :%s", flow, weight)
                if up_flow is not None:
                    flow = flow + up_flow
                weight = tf.nn.sigmoid(weight) * 1.2
                # weight = tf.nn.softplus(weight)
                flows.append(flow)
                weights.append(weight)
                trunk = tf.concat((trunk, feature), axis=3)
                trunk = hrl.network.Utils.deconv(trunk, kernel_size=3, out_channel=4, stride=2,
                                                 activation=self.nonlinear, l2=l2, var_scope="up_%d" % i)
            return {"motion": [m for m in reversed(flows)], "weight": [w for w in reversed(weights)]}

        def warp_trunk_feature(channeled_features, trunk, flow):
            # duplicate trunk into each channel of (feature, flow)
            flows = tf.split(flow, feature_channel_n, axis=3)
            features = [tf.split(f, feature_channel_n, axis=3) for f in channeled_features]
            features = [[features[i][j] for i in range(len(channeled_features))] for j in range(feature_channel_n)]
            out = []
            for f, fl in zip(features, flows):
                logging.warning("channel shape:, flow shape:%s", fl.shape)
                out.append(frame_trans(tf.concat(f+[trunk], axis=3), fl))
            return out

        def warp_channels(image_channel, flow, split_channel=True):
            if split_channel:
                image_channels = tf.split(image_channel, feature_channel_n, axis=3)
            else:
                image_channels = [image_channel] * feature_channel_n
            flows = tf.split(flow, feature_channel_n, axis=3)
            out = []
            for c, f in zip(image_channels, flows):
                logging.warning("channel shape:%s, flow shape:%s", c.shape, f.shape)
                out.append(frame_trans(c, f))
            return out
            # return [frame_trans(c, f) for c, f in zip(image_channels, flows)]

        def combine_channels(image_channel, flow, weight):
            weights = tf.split(weight, feature_channel_n, axis=3)
            warpped_channels = warp_channels(image_channel, flow)
            channels = [c * w for c, w in zip(warpped_channels, weights)]
            return tf.add_n(channels)

        def decoder_multiflow(inputs):
            input_goal = inputs[0]
            input_image = inputs[1]
            image_hw = input_image.shape.as_list()[1:3]
            with tf.name_scope("feature_extractor"):
                feature_tower = feature_extractor(input_image)["features"]
            with tf.name_scope("fcn"):
                segment = fcn(feature_tower, input_image)
            reverse_feature_tower, segment_class = segment["feature"], segment["class"]
            # from single image to segmented image
            image_channel = tf.tile(input_image, [1, 1, 1, feature_channel_n])
            segment_channel = tf.reshape(segment_class, [-1, image_hw[0], image_hw[1], feature_channel_n, 1])
            segment_channel = tf.reshape(
                tf.tile(segment_channel, [1, 1, 1, 1, 3]),
                [-1, image_hw[0], image_hw[1], 3 * feature_channel_n]
            )
            image_channel = image_channel * segment_channel
            with tf.name_scope("flow"):
                # use resized segment_class, instead of reverse_feature_tower
                resized_segment = [segment_class]
                for i in range(len(reverse_feature_tower) - 1):
                    resized_segment.append(
                        tf.image.resize_images(segment_class,
                                               (image_hw[0] / (2**(i+1)), image_hw[1] / (2**(i+1)))
                                               )
                    )

                # pwc = pwc_resflow_multi(input_goal, reverse_feature_tower, input_image)
                # pwc = pwc_resflow_multi(input_goal, [s for s in reversed(resized_segment)], input_image)
                pwc = pwc_resflow_multi_simple(input_goal, [s for s in reversed(resized_segment)], input_image, separate_kernel=True)
            motions, weights = pwc["motion"], pwc["weight"]
            with tf.name_scope("combine_image"):
                next_image = combine_channels(image_channel, motions[0], weights[0])
            sub_next_images = []
            if len(motions) > 1:
                for i in range(len(motions)-1):
                    motion, weight = motions[i+1], weights[i+1]
                    sub_image_channel = tf.image.resize_images(image_channel, motion.shape.as_list()[1:3])
                    with tf.name_scope("combine_image%d" % i):
                        sub_next_images.append(combine_channels(sub_image_channel, motion, weight))
            out = {"next_frame": next_image,
                   "flow": motions[0]}
            for i in range(len(sub_next_images)):
                out["frame_%d" % (2**(i + 1))] = sub_next_images[i]
            out["image_channel"] = image_channel
            return out
        return decoder_multiflow

    def create_env_upsample_little(self):
        def create_env_upsample_little(inputs):
            l2 = 1e-7
            input_state = inputs[0]
            input_state = tf.squeeze(tf.stack(input_state), axis=0)
        
            input_action = inputs[1]
            input_action = tf.one_hot(indices=input_action, depth=self.dim_action, on_value=1.0, off_value=0.0, axis=-1)
            input_action_tiled = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, self.dim_action]),
                                                        [((((self.dim_observation[0] + 1) / 2 + 1) / 2 + 1) / 2 + 1) / 2,
                                                         ((((self.dim_observation[1] + 1) / 2 + 1) / 2 + 1) / 2 + 1) / 2])
        
            conv_1 = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_1")
        
            conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                               shape=[(64, 4, 2)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_2")
        
            conv_3 = hrl.utils.Network.conv2ds(conv_2,
                                               shape=[(64, 3, 2)],
                                               out_flatten=True,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_3")
        
            fc_1 = hrl.utils.Network.layer_fcs(conv_3, [], 64 * 5 * 5,
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2,
                                               var_scope="fc_1")
        
            # concat_action = tf.concat([conv_4, input_action_tiled], axis=3)
            fc_action = hrl.utils.Network.layer_fcs(tf.to_float(input_action), [], 64 * 5 * 5,
                                                    activation_hidden=self.nonlinear,
                                                    activation_out=self.nonlinear,
                                                    l2=l2,
                                                    var_scope="fc_action")
        
            concat = tf.multiply(fc_1, fc_action)
        
            fc_out = hrl.utils.Network.layer_fcs(concat, [64 * 5 * 5], 64 * 5 * 5,
                                                 activation_hidden=self.nonlinear,
                                                 activation_out=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="fc_out")
        
            # reward
            reward = hrl.utils.Network.layer_fcs(fc_out, [256], 1,
                                                 activation_hidden=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="reward")
            reward = tf.squeeze(reward, axis=1)
        
            # next_state
            twoD_out = tf.reshape(fc_out, [-1, 64, 5, 5])
        
            conv_5 = hrl.utils.Network.conv2ds(twoD_out,
                                               shape=[(32, 3, 1)],
                                               out_flatten=False,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="conv_5")
        
            up_1 = tf.image.resize_images(conv_5,
                                          [((self.dim_observation[0] + 3) / 4 + 1) / 2, ((self.dim_observation[1] + 3) / 4 + 1) / 2])
        
            concat_1 = tf.concat([conv_2, up_1], axis=3)
            concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                 shape=[(32, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_1")
        
            up_2 = tf.image.resize_images(concat_1, [(self.dim_observation[0] + 3) / 4, (self.dim_observation[1] + 3) / 4])
        
            concat_2 = tf.concat([conv_1, up_2], axis=3)
            concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                 shape=[(64, 4, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_2")
        
            up_3 = tf.image.resize_images(concat_2, [self.dim_observation[0], self.dim_observation[1]])
        
            concat_3 = tf.concat([input_state, up_3], axis=3)
            concat_3 = hrl.utils.Network.conv2ds(concat_3,
                                                 shape=[(64, 3, 1)],
                                                 out_flatten=False,
                                                 activation=self.nonlinear,
                                                 l2=l2,
                                                 var_scope="concat_3")
        
            next_state = hrl.utils.Network.conv2ds(concat_3,
                                                   shape=[(3, 3, 1)],
                                                   out_flatten=False,
                                                   activation=self.nonlinear,
                                                   l2=l2,
                                                   var_scope="next_state")
        
            return {"next_state": next_state, "reward": reward}
        return create_env_upsample_little

    def create_encoder(self):
        def create_encoder(inputs):
            l2 = 1e-7
            input_rollout_states = inputs[0]
            input_reward = inputs[1]

            input_concat = tf.concat([input_rollout_states, input_reward], axis=-1)
        
            re = hrl.utils.Network.layer_fcs(input_concat, [self.dim_se], self.dim_se,
                                             activation_hidden=self.nonlinear,
                                             activation_out=self.nonlinear,
                                             l2=l2,
                                             var_scope="re")
            return {"re": re}
        return create_encoder

    def create_encoder_OB(self):
        def create_encoder(inputs):
            l2 = 1e-7
            input_rollout_states = inputs[0]
            input_reward = inputs[1]

            rse = hrl.utils.Network.conv2ds(input_rollout_states,
                                            shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                            out_flatten=True,
                                            activation=self.nonlinear,
                                            l2=l2,
                                            var_scope="rse")

            re_conv = hrl.utils.Network.layer_fcs(rse, [], 200,
                                                  activation_hidden=self.nonlinear,
                                                  activation_out=self.nonlinear,
                                                  l2=l2,
                                                  var_scope="re_conv")

            rewardNstates = tf.concat([re_conv, input_reward], axis=-1)

            re = hrl.utils.Network.layer_fcs(rewardNstates, [], self.dim_se,
                                             activation_hidden=self.nonlinear,
                                             activation_out=self.nonlinear,
                                             l2=l2,
                                             var_scope="re")

            return {"re": re}

        return create_encoder

    def create_env_upsample_fc(self):
        def create_env_upsample_fc(inputs):
            l2 = 1e-7
            input_state = inputs[0]
            # input_state = tf.squeeze(tf.stack(input_state), axis=0)

            input_action = inputs[1]
            # input_action = tf.one_hot(indices=input_action, depth=self.dim_action, on_value=1.0, off_value=0.0, axis=-1)
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

            fc_1 = hrl.utils.Network.layer_fcs(conv_3, [], 64 * 5 * 5,
                                               activation_hidden=tf.nn.relu,
                                               activation_out=tf.nn.relu,
                                               l2=l2,
                                               var_scope="fc_1")

            # concat_action = tf.concat([conv_4, input_action_tiled], axis=3)
            fc_action = hrl.utils.Network.layer_fcs(input_action, [], 64 * 5 * 5,
                                                    activation_hidden=tf.nn.relu,
                                                    activation_out=tf.nn.relu,
                                                    l2=l2,
                                                    var_scope="fc_action")

            concat = tf.multiply(fc_1, fc_action)

            fc_out = hrl.utils.Network.layer_fcs(concat, [64 * 5 * 5], 64 * 5 * 5,
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
            twoD_out = tf.reshape(fc_out, [-1, 5, 5, 64])

            conv_5 = hrl.utils.Network.conv2ds(twoD_out,
                                               shape=[(32, 3, 1)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="conv_5")

            up_1 = tf.image.resize_images(conv_5,
                                          [((self.dim_observation[0] + 3) / 4 + 1) / 2, ((self.dim_observation[1] + 3) / 4 + 1) / 2])

            concat_1 = tf.concat([conv_2, up_1], axis=3)
            concat_1 = hrl.utils.Network.conv2ds(concat_1,
                                                 shape=[(32, 3, 1)],
                                                 out_flatten=False,
                                                 activation=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="concat_1")

            up_2 = tf.image.resize_images(concat_1, [(self.dim_observation[0] + 3) / 4, (self.dim_observation[1] + 3) / 4])

            concat_2 = tf.concat([conv_1, up_2], axis=3)
            concat_2 = hrl.utils.Network.conv2ds(concat_2,
                                                 shape=[(64, 4, 1)],
                                                 out_flatten=False,
                                                 activation=tf.nn.relu,
                                                 l2=l2,
                                                 var_scope="concat_2")

            up_3 = tf.image.resize_images(concat_2, [self.dim_observation[0], self.dim_observation[1]])

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
        return create_env_upsample_fc

    def create_env_deconv_fc(self):
        def create_env_deconv_fc(inputs):
            l2 = 1e-7
            input_state = inputs[0]
            # input_state = tf.squeeze(tf.stack(input_state), axis=0)

            input_action = inputs[1]
            # input_action = tf.one_hot(indices=input_action, depth=self.dim_action, on_value=1.0, off_value=0.0, axis=-1)
            # input_action_tiled = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
            #                                       [((((dim_observation[0]+1)/2+1)/2+1)/2+1)/2,
            #                                        ((((dim_observation[1]+1)/2+1)/2+1)/2+1)/2])

            conv_1 = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (self.chn_se_2d, 3, 2)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="conv_1")
            linear_1 = tf.contrib.layers.flatten(conv_1)

            # conv_2 = hrl.utils.Network.conv2ds(conv_1,
            #                                    shape=[(64, 4, 2)],
            #                                    out_flatten=False,
            #                                    activation=tf.nn.relu,
            #                                    l2=l2,
            #                                    var_scope="conv_2")
            #
            # conv_3 = hrl.utils.Network.conv2ds(conv_2,
            #                                    shape=[(64, 3, 2)],
            #                                    out_flatten=True,
            #                                    activation=tf.nn.relu,
            #                                    l2=l2,
            #                                    var_scope="conv_3")

            fc_1 = hrl.utils.Network.layer_fcs(linear_1, [], 64 * 5 * 5,
                                               activation_hidden=tf.nn.relu,
                                               activation_out=tf.nn.relu,
                                               l2=l2,
                                               var_scope="fc_1")

            # concat_action = tf.concat([conv_4, input_action_tiled], axis=3)
            fc_action = hrl.utils.Network.layer_fcs(input_action, [], 64 * 5 * 5,
                                                    activation_hidden=tf.nn.relu,
                                                    activation_out=tf.nn.relu,
                                                    l2=l2,
                                                    var_scope="fc_action")

            concat = tf.multiply(fc_1, fc_action)

            conv_1_shape = conv_1.shape.as_list()

            fc_out = hrl.utils.Network.layer_fcs(concat, [64 * 5 * 5], conv_1_shape[1] * conv_1_shape[2] * conv_1_shape[3],
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
            twoD_out = tf.reshape(fc_out, [-1, conv_1_shape[1], conv_1_shape[2], conv_1_shape[3]])

            next_frame_before = hrl.utils.Network.conv2ds_transpose(twoD_out,
                                                                    shape=[(self.chn_se_2d, 3, 2), (64, 4, 2),
                                                                           (32, 8, 4)],
                                                                    activation=self.nonlinear,
                                                                    l2=l2,
                                                                    var_scope="next_frame_before")

            next_frame = hrl.utils.Network.conv2ds(next_frame_before,
                                                   shape=[(3, 3, 1)],
                                                   out_flatten=False,
                                                   activation=self.nonlinear,
                                                   l2=l2,
                                                   var_scope="next_frame")

            return {"next_state": next_frame, "reward": reward}
        return create_env_deconv_fc

    def pass_decoder(self):
        def pass_decoder(inputs):
            input_frame = inputs[0]
            return {"next_frame": input_frame}
        return pass_decoder


class I2AFlow(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 episode_n=10000, learning_rate=1e-4, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4), batch_size=8):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if (f_tran and f_rollout and f_ac) is None:
            f = F(env)
            f_se = f.create_se()
            f_ac = f.create_ac()
            f_rollout = f.create_rollout()
            f_encoder = f.create_encoder()
            f_tran = f.create_transition_momentum()
            f_decoder = f.create_decoder_deform_flow()
            # f_decoder = f.decoder_multiflow()

        super(I2AFlow, self).__init__(env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)
Experiment.register(I2AFlow, "A3C with I2A for CarRacing")


class I2AFlowDriving(I2AFlow):
    def __init__(self, env=None, f_se=None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder=None,
                 episode_n=10000, learning_rate=1e-4, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4), batch_size=12):
        if env is None:
            env = ScaledFloatFrame(EnvNoOpSkipping(
                env=EnvRewardVec2Scalar(FrameStack(Downsample(DrSimDecisionK8S(), dst_size=(128, 128)), 4)),
                n_skip=6, gamma=discount_factor, if_random_phase=True
            ))

            # env = DrSimDecisionK8S()
            # env = RemapFrame(env, (175, 175), (128, 128), (64, 64), 0.5)
        super(I2AFlowDriving, self).__init__(env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, episode_n,
                                             learning_rate, discount_factor, entropy, batch_size)
Experiment.register(I2AFlowDriving, "A3C with I2A for Driving Simulator")


class OTDQNModelCar(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=16000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, sampler_creator=None, asynchronous=False, save_image_interval=10000, state_size=256,
                 with_momentum=True, curriculum=[1, 3, 5], skip_step=[500000, 1000000]):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if f_se is None:
            f = F(env, state_size)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder()

        super(OTDQNModelCar, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum,
                                            skip_step, sampler_creator, asynchronous, save_image_interval,
                                            with_momentum=with_momentum)
Experiment.register(OTDQNModelCar, "transition model with dqn, for CarRacing")


class OTDQNModelCar_state(OTDQNModelCar):
    def __init__(self, env=None, f_create_q=None, f_se=None, f_transition=None, f_decoder=None, with_momentum=False,
                 state_size=1600):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if f_se is None:
            f = F(env, state_size)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition()
            f_decoder = f.create_decoder()

        super(OTDQNModelCar_state, self).__init__(env=env, f_create_q=f_create_q, f_se=f_se, f_transition=f_transition,
                                                  f_decoder=f_decoder, with_momentum=with_momentum)
Experiment.register(OTDQNModelCar_state, "state transition model with dqn, for CarRacing")


class OTDQNModelCar_state_256(OTDQNModelCar_state):
    def __init__(self, state_size=256):
        super(OTDQNModelCar_state_256, self).__init__(state_size=state_size)
Experiment.register(OTDQNModelCar_state_256, "256 state transition model with dqn, for CarRacing")


class OTDQNModelCar_mom_1600(OTDQNModelCar):
    def __init__(self, state_size=1600, with_momentum=True):
        super(OTDQNModelCar_mom_1600, self).__init__(state_size=state_size, with_momentum=with_momentum)
Experiment.register(OTDQNModelCar_mom_1600, "Hidden state with 1600 size in transition model with dqn, for CarRacing")


class OTDQNModelCar_goal_256(OTDQNModelCar):
    def __init__(self, with_momentum=False):
        super(OTDQNModelCar_goal_256, self).__init__(with_momentum=with_momentum)
Experiment.register(OTDQNModelCar_goal_256, "goal 256 in transition model with dqn, for CarRacing")


class OTDQNModelCar_goal(OTDQNModelCar_mom_1600):
    def __init__(self, with_momentum=False):
        super(OTDQNModelCar_goal, self).__init__(with_momentum=with_momentum)
Experiment.register(OTDQNModelCar_goal, "goal in transition model with dqn, for CarRacing")

class OTDQNModelCar_mom_decoder(OTDQNModelCar):
    def __init__(self, env=None, episode_n=16000, f_create_q=None, f_se=None, f_transition=None, f_decoder=None):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if f_se is None:
            f = F(env, 256)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder_deconv()

        super(OTDQNModelCar_mom_decoder, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder)
Experiment.register(OTDQNModelCar_mom_decoder, "Hidden state with 1600 size in transition model with dqn, for CarRacing")


class OTDQN_ob(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=160000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, sampler_creator=None, asynchronous=False, save_image_interval=10000, with_ob=True,
                 curriculum=[1, 3, 5], skip_step=[500000, 1000000]):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_env_upsample_fc()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.pass_decoder()
        super(OTDQN_ob, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum, skip_step,
                                            sampler_creator, asynchronous, save_image_interval, with_ob)
Experiment.register(OTDQN_ob, "Old traditional env model with dqn, for CarRacing")


class OTDQN_ob_decoder(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=160000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, curriculum=[1, 3, 5], skip_step=[500000, 1000000], sampler_creator=None,
                 asynchronous=False, save_image_interval=10000, with_ob=True):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_env_deconv_fc()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.pass_decoder()
        super(OTDQN_ob_decoder, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum, skip_step,
                                            sampler_creator, asynchronous, save_image_interval, with_ob)
Experiment.register(OTDQN_ob_decoder, "Old traditional env model with dqn, for CarRacing")


class OTDQNModelDriving(OTDQNModelCar):
    def __init__(self, env=None, episode_n=20000, f_create_q=None, f_se=None, f_transition=None, f_decoder=None,
                 lower_weight=1.0, upper_weight=1.0, rollout_depth=5, discount_factor=0.99, ddqn=False,
                 target_sync_interval=100, target_sync_rate=1.0, greedy_epsilon=0.1, network_optimizer=None,
                 max_gradient=10.0, update_interval=4, replay_size=10000, batch_size=10, sampler_creator=None,
                 state_size=1600, asynchronous=True):
        if env is None:
            env = ScaledFloatFrame(EnvNoOpSkipping(
                        env=EnvRewardVec2Scalar(
                            FrameStack(
                                Downsample(
                                    DrSimDecisionK8STopView()
                                   , dst_size=(128, 128)
                                )
                                , 4
                            )
                        ),
                        n_skip=6, gamma=0.99, if_random_phase=True
                    )
            )
        # if sampler_creator is None:
        #     max_traj_length = 200
        #
        #     def create_sample(args):
        #         bucket_size = 8
        #         traj_count = replay_size / max_traj_length
        #         bucket_count = traj_count / bucket_size
        #         active_bucket = 4
        #         ratio = 1.0 * active_bucket / bucket_count
        #         transition_epoch = 8
        #         trajectory_epoch = transition_epoch * max_traj_length
        #         memory = BigPlayback(
        #             bucket_cls=Playback,
        #             bucket_size=bucket_size,
        #             max_sample_epoch=trajectory_epoch,
        #             capacity=traj_count,
        #             active_ratio=ratio,
        #             cache_path=os.sep.join([args.logdir, "cache", str(args.index)])
        #         )
        #         sampler = sampling.TruncateTrajectorySampler2(memory, replay_size / max_traj_length, max_traj_length,
        #                                                       batch_size=1, trajectory_length=batch_size,
        #                                                       interval=update_interval)
        #         return sampler
        #     sampler_creator = create_sample
        super(OTDQNModelDriving, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                                upper_weight, rollout_depth, discount_factor, ddqn,
                                                target_sync_interval, target_sync_rate, greedy_epsilon,
                                                network_optimizer, max_gradient, update_interval, replay_size,
                                                batch_size, sampler_creator, asynchronous, state_size=state_size)
Experiment.register(OTDQNModelDriving, "transition model with dqn, for k8s driving env")


if __name__ == '__main__':
    Experiment.main()