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
            # env = Downsample(env, length_factor=2.0)
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
            env = Downsample(env, length_factor=2.0)
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


class Freeway(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_tran=None, f_decoder=None, f_rollout=None, f_encoder = None,
                 episode_n=10000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 2e-2, 5e-3), batch_size=32, policy_with_iaa=False,
                 compute_with_diff=False, with_momentum=True, dynamic_rollout=[1, 3, 5],
                 dynamic_skip_step=[10000, 20000]):
        if env is None:
            env = gym.make('Freeway-v0')
            # env = Downsample(env, length_factor=2.0)
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
                                      dynamic_skip_step)


class Freeway_mom(Freeway):
    def __init__(self):
        super(Freeway_mom, self).__init__()
Experiment.register(Freeway_mom, "Momentum exp for Freeway")


class Freeway_mom_half(Freeway):
    def __init__(self, env=None, dynamic_skip_step=[30000, 60000]):
        if env is None:
            env = gym.make('Freeway-v0')
            env = Downsample(env, length_factor=2.0)
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
            env = Downsample(env, length_factor=2.0)
            env = ScaledFloatFrame(env)
            env = MaxAndSkipEnv(env, skip=4, max_len=1)
            env = FrameStack(env, k=4)
        super(Freeway_mom_I2A_half, self).__init__(env=env, policy_with_iaa=policy_with_iaa, dynamic_skip_step=dynamic_skip_step)
Experiment.register(Freeway_mom_I2A_half, "I2A with Momentum exp for Freeway with half input")


if __name__ == '__main__':
    Experiment.main()