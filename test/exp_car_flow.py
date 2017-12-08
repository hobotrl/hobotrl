#
# -*- coding: utf-8 -*-


import sys

from hobotrl.async import AsynchronousAgent
from hobotrl.utils import CappedLinear

sys.path.append(".")

from exp_car import *
from hobotrl.tf_dependent.ops import frame_trans, CoordUtil
from playground.initialD.ros_environments.clients import DrSimDecisionK8S
from hobotrl.environments.environments import RemapFrame
from playground.ot_model import OTModel


class F(object):
    def __init__(self, env):
        super(F, self).__init__()
        self._env = env
        self.dim_action = self._env.action_space.n
        self.dim_observation = self._env.observation_space.shape
        self.chn_se_2d = 32
        self.dim_se = 256
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
        
            se_linear = hrl.utils.Network.layer_fcs(se_conv, [256], self.dim_se,
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
        
            fc_action = hrl.utils.Network.layer_fcs(input_action, [], 256,
                                                    activation_hidden=self.nonlinear,
                                                    activation_out=self.nonlinear,
                                                    l2=l2,
                                                    var_scope="fc_action")
        
            concat = tf.multiply(input_state, fc_action)
            # concat = tf.concat([input_state, fc_action], axis=-1)
        
            fc_out = hrl.utils.Network.layer_fcs(concat, [], 256,
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
            next_state = hrl.utils.Network.layer_fcs(fc_out, [256], 256,
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
        
            fc_action = hrl.utils.Network.layer_fcs(input_action, [], 256,
                                                    activation_hidden=self.nonlinear,
                                                    activation_out=self.nonlinear,
                                                    l2=l2,
                                                    var_scope="fc_action")
        
            concat = tf.multiply(input_state, fc_action)
        
            fc_out = hrl.utils.Network.layer_fcs(concat, [], 256,
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
            Action_related_goal = hrl.utils.Network.layer_fcs(fc_out, [256], self.dim_se,
                                                              activation_hidden=self.nonlinear,
                                                              activation_out=None,
                                                              l2=l2,
                                                              var_scope="TC")
        
            Action_unrelated_goal = hrl.utils.Network.layer_fcs(input_state, [256], self.dim_se,
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
                                               (2 * 256 / conv_3_shape[1] / conv_3_shape[2]),
                                               activation_hidden=self.nonlinear,
                                               activation_out=self.nonlinear,
                                               l2=l2,
                                               var_scope="fc_goal")
        
            twoD_out = tf.reshape(fc_1, [-1, conv_3_shape[1], conv_3_shape[2],
                                         2 * 256 / conv_3_shape[1] / conv_3_shape[2]])
        
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
            coord_y, coord_x = CoordUtil.get_coord_tensor(n, h, w)
            # normalize to [-1.0, 1.0]
            coord_y = tf.to_float(coord_y) / tf.to_float(h) * 2.0 - 1.0
            coord_x = tf.to_float(coord_x) / tf.to_float(w) * 2.0 - 1.0
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
            input_argu = inputs[0]
            # input_reward = inputs[1]
            #
            # rse = hrl.utils.Network.conv2ds(input_state,
            #                                 shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            #                                 out_flatten=True,
            #                                 activation=self.nonlinear,
            #                                 l2=l2,
            #                                 var_scope="rse")
            #
            # re_conv = hrl.utils.Network.layer_fcs(rse, [], 200,
            #                                       activation_hidden=self.nonlinear,
            #                                       activation_out=self.nonlinear,
            #                                       l2=l2,
            #                                       var_scope="re_conv")
            #
            # # re_conv = tf.concat([re_conv, tf.reshape(input_reward, [-1, 1])], axis=1)
            # re_conv = tf.concat([re_conv, input_reward], axis=1)
        
            re = hrl.utils.Network.layer_fcs(input_argu, [256], 256,
                                             activation_hidden=self.nonlinear,
                                             activation_out=self.nonlinear,
                                             l2=l2,
                                             var_scope="re")
        
            return {"re": re}
        return create_encoder


class I2AFlow(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 episode_n=10000, learning_rate=1e-4, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4), batch_size=12):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if (f_tran and f_rollout and f_ac) is None:
            f = F(env)
            f_se = f.create_se()
            f_ac = f.create_ac()
            # f_env = create_env_upsample_little
            f_rollout = f.create_rollout()
            f_encoder = f.create_encoder()
            f_tran = f.create_transition_momentum()
            # f_decoder = create_decoder
            f_decoder = f.create_decoder_deform_flow()

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
    def __init__(self, env=None, episode_n=10000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, sampler_creator=None, asynchronous=False):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)
        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            f_decoder = f.create_decoder_deform_flow()
        super(OTDQNModelCar, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, sampler_creator, asynchronous)
Experiment.register(OTDQNModelCar, "transition model with dqn, for CarRacing")


class OTDQNModelDriving(OTDQNModelCar):
    def __init__(self, env=None, episode_n=10000, f_create_q=None, f_se=None, f_transition=None, f_decoder=None,
                 lower_weight=1.0, upper_weight=1.0, rollout_depth=5, discount_factor=0.99, ddqn=False,
                 target_sync_interval=100, target_sync_rate=1.0, greedy_epsilon=0.1, network_optimizer=None,
                 max_gradient=10.0, update_interval=4, replay_size=100000, batch_size=12, sampler_creator=None,
                 asynchronous=True):
        if env is None:
            env = ScaledFloatFrame(EnvNoOpSkipping(
                        env=EnvRewardVec2Scalar(FrameStack(Downsample(DrSimDecisionK8S(), dst_size=(128, 128)), 4)),
                        n_skip=6, gamma=0.99, if_random_phase=True
                    ))
        if sampler_creator is None:
            max_traj_length = 200

            def create_sample(args):
                bucket_size = 8
                traj_count = replay_size / max_traj_length
                bucket_count = traj_count / bucket_size
                active_bucket = 4
                ratio = 1.0 * active_bucket / bucket_count
                transition_epoch = 8
                trajectory_epoch = transition_epoch * max_traj_length
                memory = BigPlayback(
                    bucket_cls=Playback,
                    bucket_size=bucket_size,
                    max_sample_epoch=trajectory_epoch,
                    capacity=traj_count,
                    active_ratio=ratio,
                    cache_path=os.sep.join([args.logdir, "cache", str(args.index)])
                )
                sampler = sampling.TruncateTrajectorySampler2(memory, replay_size / max_traj_length, max_traj_length,
                                                              batch_size=1, trajectory_length=batch_size,
                                                              interval=update_interval)
                return sampler
            sampler_creator = create_sample
        super(OTDQNModelDriving, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                                upper_weight, rollout_depth, discount_factor, ddqn,
                                                target_sync_interval, target_sync_rate, greedy_epsilon,
                                                network_optimizer, max_gradient, update_interval, replay_size,
                                                batch_size, sampler_creator, asynchronous)
Experiment.register(OTDQNModelDriving, "transition model with dqn, for k8s driving env")


if __name__ == '__main__':
    Experiment.main()