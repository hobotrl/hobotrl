#
# -*- coding: utf-8 -*-


import sys
sys.path.append(".")
import logging
import numpy as np
import gym
import cv2
import matplotlib.colors as colors
from exp_algorithms import *
import hobotrl.environments as envs


class ProcessFrame96H(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame96H, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 1))

    def _observation(self, obs):
        return ProcessFrame96H.process(obs)

    @staticmethod
    def process(frame):
        if list(frame.shape) == [96, 96, 3]:
            pass
        else:
            assert False, "Unknown resolution."
        img = frame
        img = colors.rgb_to_hsv(img / 255.0)
        img = np.transpose(img, axes=[2, 0, 1])[0]
        img = (img * 255).astype(np.uint8).reshape((96, 96, 1))
        return img


class CarContinuousWrapper(gym.Wrapper):

    def __init__(self, env):
        super(CarContinuousWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, [2])

    def _step(self, action):
        env_action = np.zeros(3)
        env_action[0] = action[0]
        if action[1] > 0:
            env_action[1], env_action[2] = action[1], 0
        else:
            env_action[1], env_action[2] = 0, -action[1]
        return self.env.step(env_action)


class CarGrassWrapper(gym.Wrapper):

    def __init__(self, env, grass_penalty=0.5):
        super(CarGrassWrapper, self).__init__(env)
        self.grass_penalty = grass_penalty

    def _step(self, action):

        ob, reward, done, info = self.env.step(action)
        if (ob[71:76, 47:49, 0] > 200).all():  # red car visible
            front = (ob[70, 47:49, 1] > 200).all()
            back = (ob[76, 47:49, 1] > 200).all()
            left = (ob[71:74, 46, 1] > 200).all()
            right = (ob[71:74, 49, 1] > 200).all()
            if front and back and left and right:
                reward -= self.grass_penalty
        return ob, reward, done, info


def wrap_car(env):
    """Apply a common set of wrappers for Box2d games."""
    env = CarGrassWrapper(env, grass_penalty=0.5)
    env = CarContinuousWrapper(env)
    env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
    env = envs.FrameStack(env, 4)
    env = envs.ScaledRewards(env, 0.1)
    env = envs.ScaledFloatFrame(env)
    env = envs.AugmentEnvWrapper(env, reward_decay=0.99)
    return env


class A3CCarExp(ACOOExperimentCon):
    def __init__(self, env, f_create_net=None,
                 episode_n=1000,
                 reward_decay=0.99,
                 entropy_scale=0.01,
                 on_batch_size=32,
                 off_batch_size=32,
                 off_interval=0,
                 sync_interval=1000,
                 replay_size=128,
                 prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-3),
                 l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(5e-5), ddqn=False, aux_r=False, aux_d=False):

        def create_ac_car(input_state, num_action, **kwargs):
            se = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se")


            v = hrl.utils.Network.layer_fcs(se, [256], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="v")
            v = tf.squeeze(v, axis=1)

            pi_mean = hrl.utils.Network.layer_fcs(se, [256], num_action,
                                             activation_hidden=tf.nn.relu,
                                             activation_out=None,
                                             l2=l2,
                                             var_scope="pi_mean")
            pi_mean = tf.nn.tanh(pi_mean / 4.0)

            pi_stddev = hrl.utils.Network.layer_fcs(se, [256], 1,
                                                  activation_hidden=tf.nn.relu,
                                                  activation_out=None,
                                                  l2=l2,
                                                  var_scope="pi_stddev")
            pi_stddev = 4.0 * tf.nn.sigmoid(pi_stddev / 4.0)

            r = hrl.utils.Network.layer_fcs(se, [256], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="r")
            r = tf.squeeze(r, axis=1)

            return {"pi_mean": pi_mean, "pi_stddev": pi_stddev, "v": v, "se": se, "r": r}
        if f_create_net is None:
            f_create_net = create_ac_car
        logging.warning("before super(A3CCarExp, self).__init__")
        super(A3CCarExp, self).__init__(env, f_create_net, episode_n, reward_decay, entropy_scale, on_batch_size,
                                        off_batch_size, off_interval, sync_interval, replay_size, prob_min, entropy,
                                        l2, optimizer_ctor, ddqn, aux_r, aux_d)


class A3CCarRacing(A3CCarExp):
    def __init__(self):
        env = gym.make("CarRacing-v0")
        env = wrap_car(env)
        super(A3CCarRacing, self).__init__(env)

Experiment.register(A3CCarRacing, "Continuous A3C for CarRacing")


class A3CPendulumExp(ACOOExperimentCon):
    def __init__(self, env, f_create_net=None,
                 episode_n=10000,
                 reward_decay=0.99,
                 entropy_scale=0.05,
                 on_batch_size=32,
                 off_batch_size=32,
                 off_interval=0,
                 sync_interval=1000,
                 replay_size=128,
                 prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-3),
                 l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(6e-5), ddqn=False, aux_r=False, aux_d=False):

        def create_ac_pendulum(input_state, num_action, **kwargs):
            se = hrl.utils.Network.layer_fcs(input_state, [200], 200,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=tf.nn.relu,
                                            l2=l2,
                                            var_scope="se")

            v = hrl.utils.Network.layer_fcs(se, [200], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="v")
            v = tf.squeeze(v, axis=1)

            pi_mean = hrl.utils.Network.layer_fcs(se, [200], num_action,
                                             activation_hidden=tf.nn.relu,
                                             activation_out=tf.nn.tanh,
                                             l2=l2,
                                             var_scope="pi_mean")

            pi_stddev = hrl.utils.Network.layer_fcs(se, [200], num_action,
                                                  activation_hidden=tf.nn.relu,
                                                  activation_out=tf.nn.softplus,
                                                  l2=l2,
                                                  var_scope="pi_stddev")

            r = hrl.utils.Network.layer_fcs(se, [200], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="r")
            r = tf.squeeze(r, axis=1)

            return {"pi_mean": pi_mean, "pi_stddev": pi_stddev, "v": v,
                    "se": se, "r": r}

        if f_create_net is None:
            f_create_net = create_ac_pendulum
        logging.warning("before super(A3CPendulumExp, self).__init__")
        super(A3CPendulumExp, self).__init__(env, f_create_net, episode_n, reward_decay, entropy_scale, on_batch_size,
                                             off_batch_size, off_interval, sync_interval, replay_size, prob_min,
                                             entropy, l2, optimizer_ctor, ddqn, aux_r, aux_d)


class A3CPendulum(A3CPendulumExp):
    def __init__(self):
        env = gym.make("Pendulum-v0")
        env = hrl.envs.AugmentEnvWrapper(
            env, reward_decay=0.9, reward_scale=0.1,
            action_limit=np.asarray([env.action_space.low, env.action_space.high])
        )
        super(A3CPendulum, self).__init__(env)
Experiment.register(A3CPendulum, "Continuous A3C for Pendulum")


if __name__ == '__main__':
    Experiment.main()
