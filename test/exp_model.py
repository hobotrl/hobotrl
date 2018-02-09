import logging

import sys
sys.path.append(".")

from hobotrl.network import Utils
from exp_algorithms import *
from hobotrl.playback import Playback
from hobotrl.playback import BigPlayback
from car import *

class Model(TransitionModel):
    def __init__(self, env=None, f_se=None, f_transition=None, f_decoder=None,
                 network_optimizer=None,
                 max_gradient=10.0,
                 rollout_depth=5,
                 update_interval=4,
                 replay_size=100000,
                 batch_size=16,
                 curriculum=[1, 3, 5],
                 skip_step=[500000, 1000000],
                 sampler_creator=None,
                 save_image_interval=10000,
                 with_ob=False,
                 with_momentum=True,
                 with_goal=True):
        # if env is None:
        #     env = gym.make('Freeway-v0')
        #     env = Downsample(env, dst_size=[96, 96])
        #     env = ScaledFloatFrame(env)
        #     env = MaxAndSkipEnv(env, skip=4, max_len=1)
        #     env = FrameStack(env, k=4)

        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3)

        if f_se is None:
            l2 = 1e-7
            self.nonlinear = tf.nn.elu
            self.dim_se = 256

            def create_se(inputs):
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(8, 8, 4), (16, 4, 2), (32, 3, 2)],
                                               out_flatten=True,
                                               activation=self.nonlinear,
                                               l2=l2,
                                               var_scope="se")
                se_linear = hrl.utils.Network.layer_fcs(se, [], 256,
                                                        activation_hidden=self.nonlinear,
                                                        activation_out=self.nonlinear,
                                                        l2=l2,
                                                        var_scope="se_linear")
                return {"se": se_linear}

            def create_transition(inputs):
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

            def create_transition_momentum(inputs):
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

            def create_decoder(inputs):
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

            f_se = create_se
            f_transition = create_transition_momentum
            f_decoder = create_decoder

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
                                                              interval=update_interval, update_on=False)
                return sampler
            sampler_creator = create_sample

        super(Model, self).__init__(env, f_se, f_transition, f_decoder, rollout_depth, network_optimizer, max_gradient,
                                    update_interval, replay_size, batch_size, curriculum, skip_step, sampler_creator,
                                    save_image_interval, with_ob, with_momentum, with_goal)


Experiment.register(Model, "only train the MTM")


if __name__ == '__main__':
    Experiment.main()
