#
# -*- coding: utf-8 -*-


import sys
import logging

sys.path.append(".")
from exp_car_flow import *
import hobotrl as hrl

class Freeway_search(hrl.experiment.GridSearch):
    def __init__(self):
        super(Freeway_search, self).__init__(Freeway_A3C_half, {
            "entropy": [CappedLinear(1e6, 2e-2, 5e-3), CappedLinear(1e6, 5e-2, 5e-3)],
            "batch_size": [16, 32],
            "episode_n": [2000],
        })
Experiment.register(Freeway_search, "Hyperparam search for Freeway with A3C")


class Freeway_A3C(A3CExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=10000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 2e-2, 5e-3),
                 batch_size=32):
        if env is None:
            env = gym.make('Freeway-v0')
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        if f_create_net is None:
            dim_action = env.action_space.n

            def create_ac_car(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (32, 3, 2)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
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
            f_create_net = create_ac_car
        super(Freeway_A3C, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                              batch_size)
Experiment.register(Freeway_A3C, "A3C for Freeway")


class Freeway_A3C_half(A3CExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=10000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 2e-2, 5e-3),
                 batch_size=32):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        if f_create_net is None:
            dim_action = env.action_space.n

            def create_ac_car(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (32, 3, 2)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
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
            f_create_net = create_ac_car
        super(Freeway_A3C_half, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                              batch_size)
Experiment.register(Freeway_A3C_half, "A3C for Freeway with half input observation")


class Freeway_A3C_half_1e_4(Freeway_A3C_half):
    def __init__(self, learning_rate=1e-4):
        super(Freeway_A3C_half_1e_4, self).__init__(learning_rate=learning_rate)
Experiment.register(Freeway_A3C_half_1e_4, "A3C for Freeway with half input observation")


class Freeway(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 episode_n=10000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 2e-2, 5e-3), batch_size=32, policy_with_iaa=False,
                 compute_with_diff=False, with_momentum=True, dynamic_rollout=[1, 3, 5],
                 dynamic_skip_step=[10000, 20000], with_ob=False):
        if env is None:
            env = gym.make('Freeway-v0')
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        if (f_tran and f_rollout and f_ac) is None:
            f = F(env)
            f_se = f.create_se()
            f_ac = f.create_ac()
            f_rollout = f.create_rollout()
            f_encoder = f.create_encoder()
            f_tran = f.create_transition_momentum()
            f_decoder = f.create_decoder()

        super(Freeway, self).__init__(env, f_se, f_ac, f_tran, f_decoder, f_rollout, f_encoder, episode_n,
                                      learning_rate, discount_factor, entropy, batch_size,
                                      policy_with_iaa,
                                      compute_with_diff,
                                      with_momentum,
                                      dynamic_rollout,
                                      dynamic_skip_step,
                                      with_ob=with_ob)


class Freeway_mom(Freeway):
    def __init__(self):
        super(Freeway_mom, self).__init__()
Experiment.register(Freeway_mom, "Momentum exp for Freeway")


class Freeway_mom_half(Freeway):
    def __init__(self, env=None, dynamic_skip_step=[30000, 60000]):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        super(Freeway_mom_half, self).__init__(env=env, dynamic_skip_step=dynamic_skip_step)
Experiment.register(Freeway_mom_half, "Momentum exp for Freeway with half input")


class Freeway_mom_I2A(Freeway):
    def __init__(self, policy_with_iaa=True):
        super(Freeway_mom_I2A, self).__init__(policy_with_iaa=policy_with_iaa)
Experiment.register(Freeway_mom_I2A, "I2A with Momentum exp for Freeway")


class Freeway_mom_I2A_half(Freeway):
    def __init__(self, env=None, policy_with_iaa=True, dynamic_skip_step=[30000, 60000]):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        super(Freeway_mom_I2A_half, self).__init__(env=env, policy_with_iaa=policy_with_iaa, dynamic_skip_step=dynamic_skip_step)
Experiment.register(Freeway_mom_I2A_half, "I2A with Momentum exp for Freeway with half input")


class Freeway_ob_I2A(Freeway):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 with_ob=True):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if f_se is None:
            f = F(env)
            f_se = f.create_se()
            f_ac = f.create_ac()
            f_rollout = f.create_rollout()
            f_encoder = f.create_encoder()
            f_tran = f.create_env_upsample_fc()
            f_decoder = f.pass_decoder()

        super(Freeway_ob_I2A, self).__init__(env=env, f_se=f_se, f_ac=f_ac, f_rollout=f_rollout, f_encoder=f_encoder,
                                             f_tran=f_tran, f_decoder=f_decoder, with_ob=with_ob)
Experiment.register(Freeway_ob_I2A, "Original I2A for Freeway")


class FreewayOTDQN_mom(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=16000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, curriculum=[1, 3, 5], skip_step=[500000, 1000000],
                 sampler_creator=None, asynchronous=False, save_image_interval=10000, with_momentum=True):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if f_se is None:
            f = F(env, 256)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder()

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

        super(FreewayOTDQN_mom, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                               upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                               target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                               update_interval, replay_size, batch_size, curriculum,
                                               skip_step, sampler_creator, asynchronous, save_image_interval,
                                               with_momentum=with_momentum)
Experiment.register(FreewayOTDQN_mom, "OTDQN for Freeway with half input")


class FreewayOTDQN_goal_256(FreewayOTDQN_mom):
    def __init__(self, with_momentum=False):
        super(FreewayOTDQN_goal_256, self).__init__(with_momentum=with_momentum)
Experiment.register(FreewayOTDQN_goal_256, "goal rollout OTDQN for Freeway with half input")


class FreewayOTDQN_mom_1600(FreewayOTDQN_mom):
    def __init__(self, env=None, episode_n=16000, f_create_q=None, f_se=None, f_transition=None, f_decoder=None,
                 with_momentum=True):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if f_se is None:
            f = F(env, 1600)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder()

        super(FreewayOTDQN_mom_1600, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder,
                                                    with_momentum=with_momentum)
Experiment.register(FreewayOTDQN_mom_1600, "Hidden state size of 1600 on OTDQN for Freeway with half input")


class FreewayOTDQN_goal(FreewayOTDQN_mom_1600):
    def __init__(self, with_momentum=False):
        super(FreewayOTDQN_goal, self).__init__(with_momentum=with_momentum)
Experiment.register(FreewayOTDQN_goal, "goal rollout OTDQN for Freeway with half input")


class FreewayOTDQN_mom_decoder(FreewayOTDQN_mom):
    def __init__(self, env=None, episode_n=16000, f_create_q=None, f_se=None, f_transition=None, f_decoder=None):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)

        if f_se is None:
            f = F(env, 256)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_transition_momentum()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.create_decoder_deconv()

        super(FreewayOTDQN_mom_decoder, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder)
Experiment.register(FreewayOTDQN_mom_decoder, "Deconv decoder and hidden state size of 256 on OTDQN for Freeway with half input")


class OTDQN_ob_Freeway(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=160000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, curriculum=[1, 3, 5], skip_step=[500000, 1000000], sampler_creator=None,
                 asynchronous=False, save_image_interval=10000, with_ob=True):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_env_upsample_fc()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.pass_decoder()
        super(OTDQN_ob_Freeway, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum, skip_step,
                                            sampler_creator, asynchronous, save_image_interval, with_ob)
Experiment.register(OTDQN_ob_Freeway, "Old traditional env model with dqn, for Freeway")


class OTDQN_ob_decoder_Freeway(OTDQNModelExperiment):
    def __init__(self, env=None, episode_n=160000,
                 f_create_q=None, f_se=None, f_transition=None, f_decoder=None, lower_weight=1.0, upper_weight=1.0,
                 rollout_depth=5, discount_factor=0.99, ddqn=False, target_sync_interval=100, target_sync_rate=1.0,
                 greedy_epsilon=0.1, network_optimizer=None, max_gradient=10.0, update_interval=4, replay_size=1024,
                 batch_size=16, curriculum=[1, 3, 5], skip_step=[500000, 1000000], sampler_creator=None,
                 asynchronous=False, save_image_interval=10000, with_ob=True):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, dst_size=[96, 96])
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        if f_se is None:
            f = F(env)
            f_create_q = f.create_q()
            f_se = f.create_se()
            f_transition = f.create_env_deconv_fc()
            # f_decoder = f.decoder_multiflow()
            f_decoder = f.pass_decoder()
        super(OTDQN_ob_decoder_Freeway, self).__init__(env, episode_n, f_create_q, f_se, f_transition, f_decoder, lower_weight,
                                            upper_weight, rollout_depth, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, greedy_epsilon, network_optimizer, max_gradient,
                                            update_interval, replay_size, batch_size, curriculum, skip_step,
                                            sampler_creator, asynchronous, save_image_interval, with_ob)
Experiment.register(OTDQN_ob_decoder_Freeway, "Old traditional env model with dqn, for Freeway")


if __name__ == '__main__':
    Experiment.main()
