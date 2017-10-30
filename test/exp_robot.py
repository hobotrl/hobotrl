#
# -*- coding: utf-8 -*-


import sys
sys.path.append(".")

from OpenGL import GL
import gym
import roboschool
from exp_algorithms import *
import hobotrl.environments as envs


class A3CHumanoidContinuous(A3CExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=1000000, learning_rate=5e-5, discount_factor=0.95,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-3, 1e-4),
                 batch_size=64):
        if env is None:
            env = gym.make("RoboschoolHumanoid-v1")
            env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
            # env = ProcessFrame96H(env)
            # env = envs.FrameStack(env, 4)
            env = envs.ScaledRewards(env, 0.2)
            # env = envs.ScaledFloatFrame(env)
        if f_create_net is None:
            dim_action = env.action_space.shape[-1]

            def create_ac_car(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se = hrl.utils.Network.layer_fcs(input_state,
                                                 [256, 256, 256], 256,
                                                 activation_hidden=tf.nn.elu,
                                                 l2=l2,
                                                 var_scope="se")

                v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                activation_hidden=tf.nn.elu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                mean = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                   activation_hidden=tf.nn.elu,
                                                   activation_out=None,
                                                   l2=l2,
                                                   var_scope="mean")
                mean = tf.nn.tanh(mean / 4.0)
                stddev = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                     activation_hidden=tf.nn.elu,
                                                     # activation_out=tf.nn.softplus,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="stddev")
                stddev = 4.0 * tf.nn.sigmoid(stddev / 4.0)
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = create_ac_car
        super(A3CHumanoidContinuous, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                                    batch_size)
Experiment.register(A3CHumanoidContinuous, "continuous A3C for Robot")


class A3CAnt(A3CHumanoidContinuous):
    def __init__(self, env=None, f_create_net=None, episode_n=1000000, learning_rate=5e-5, discount_factor=0.95,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-3, 1e-4), batch_size=64):
        if env is None:
            env = gym.make("RoboschoolAnt-v1")
            env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
            # env = ProcessFrame96H(env)
            # env = envs.FrameStack(env, 4)
            env = envs.ScaledRewards(env, 0.2)

        super(A3CAnt, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy, batch_size)
Experiment.register(A3CAnt, "continuous A3C for Ant")


if __name__ == '__main__':
    Experiment.main()
