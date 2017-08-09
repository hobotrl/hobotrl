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


class CarDiscreteWrapper(gym.Wrapper):
    """
    Wraps car env into discrete action control problem
    """

    def __init__(self, env, steer_n, speed_n):
        super(CarDiscreteWrapper, self).__init__(env)
        self.steer_n, self.speed_n = steer_n, speed_n
        self.env = env
        self.action_n = steer_n * speed_n
        self.action_space = gym.spaces.discrete.Discrete(self.action_n)

    def __getattr__(self, name):
        print("getattr:", name, " @ ", id(self.env))
        if name == "action_space":
            print("getattr: action_space:", name)
            return self.action_space
        else:
            return getattr(self.env, name)

    def _step(self, action):
        action_c = self.action_d2c(action)
        next_state, reward, done, info = self.env.step(action_c)

        return next_state, reward, done, info

    def action_c2d(self, action):
        """
        continuous action to discrete action
        :param action:
        :return:
        """
        steer_i = int((action[0] - (-1.0)) / 2.0 * self.steer_n)
        steer_i = self.steer_n - 1 if steer_i >= self.steer_n else steer_i
        if abs(action[1]) > abs(action[2]):
            speed_action = action[1]
        else:
            speed_action = -action[2]
        speed_i = int((speed_action - (-1.0)) / 2.0 * self.speed_n)
        speed_i = self.speed_n - 1 if speed_i >= self.speed_n else speed_i
        return steer_i * self.speed_n + speed_i

    def action_d2c(self, action):
        steer_i = int(action / self.speed_n)
        speed_i = action % self.speed_n
        action_c = np.asarray([0., 0., 0.])
        action_c[0] = float(steer_i) / self.steer_n * 2 - 1.0 + 1.0 / self.steer_n
        speed_c = float(speed_i) / self.speed_n * 2 - 1.0 + 1.0 / self.speed_n
        if speed_c >= 0:
            action_c[1], action_c[2] = speed_c, 0
        else:
            action_c[1], action_c[2] = 0, -speed_c
        return action_c


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


def wrap_car(env, steer_n, speed_n):
    """Apply a common set of wrappers for Atari games."""
    env = CarDiscreteWrapper(env, steer_n, speed_n)
    env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
    # env = ProcessFrame96H(env)
    env = envs.FrameStack(env, 4)
    env = envs.ScaledRewards(env, 0.1)
    env = envs.ScaledFloatFrame(env)
    return env


class A3CCarExp(ACOOExperiment):
    def __init__(self, env, f_create_net=None,
                 episode_n=10000,
                 reward_decay=0.99,
                 on_batch_size=32,
                 off_batch_size=32,
                 off_interval=0,
                 sync_interval=1000,
                 replay_size=128,
                 prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-3),
                 l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(1e-4), ddqn=False, aux_r=False, aux_d=False):

        def create_ac_car(input_state, num_action, **kwargs):
            se_v = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se_v")

            q = hrl.utils.Network.layer_fcs(se, [256], num_action,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="q")
            pi = hrl.utils.Network.layer_fcs(se, [256], num_action,
                                             activation_hidden=tf.nn.relu,
                                             # activation_out=tf.nn.softplus,
                                             l2=l2,
                                             var_scope="pi")

            pi = tf.nn.softmax(pi)
            # pi = pi + prob_min
            # pi = pi / tf.reduce_sum(pi, axis=-1, keep_dims=True)
            r = hrl.utils.Network.layer_fcs(se, [256], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="r")

            return {"pi": pi, "q": q, "se": se, "r": r}
        if f_create_net is None:
            f_create_net = create_ac_car
        logging.warning("before super(A3CCarExp, self).__init__")
        super(A3CCarExp, self).__init__(env, f_create_net, episode_n, reward_decay, on_batch_size, off_batch_size,
                                     off_interval, sync_interval, replay_size, prob_min, entropy, l2, optimizer_ctor,
                                     ddqn, aux_r, aux_d)


class A3CCarDiscrete(A3CCarExp):
    def __init__(self):
        env = gym.make("CarRacing-v0")
        env = wrap_car(env, 3, 3)
        super(A3CCarDiscrete, self).__init__(env)

Experiment.register(A3CCarDiscrete, "discrete A3C for CarRacing")


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


class A3CPendulumExp(ACOOExperimentCon):
    def __init__(self, env, f_create_net=None,
                 episode_n=10000,
                 reward_decay=0.99,
                 on_batch_size=32,
                 off_batch_size=32,
                 off_interval=0,
                 sync_interval=1000,
                 replay_size=128,
                 prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-3),
                 l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(1e-4), ddqn=False, aux_r=False, aux_d=False):

        def create_ac_pendulum(input_state, num_action, **kwargs):
            se_v = hrl.utils.Network.layer_fcs(input_state, [200], 100,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=tf.nn.relu,
                                            l2=l2,
                                            var_scope="se_v")

            se_pi = hrl.utils.Network.layer_fcs(input_state, [200], 100,
                                               activation_hidden=tf.nn.relu,
                                               activation_out=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se_pi")

            v = hrl.utils.Network.layer_fcs(se_v, [50], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="v")
            v = tf.squeeze(v, axis=1)

            pi_mean = hrl.utils.Network.layer_fcs(se_pi, [50], num_action,
                                             activation_out=tf.nn.tanh,
                                             l2=l2,
                                             var_scope="pi_mean")

            pi_stddev = hrl.utils.Network.layer_fcs(se_pi, [50], num_action,
                                                  activation_out=tf.nn.softplus,
                                                  l2=l2,
                                                  var_scope="pi_stddev")

            r = hrl.utils.Network.layer_fcs(se_v, [50], 1,
                                            activation_hidden=tf.nn.relu,
                                            l2=l2,
                                            var_scope="r")
            r = tf.squeeze(r, axis=1)

            return {"pi_mean": pi_mean, "pi_stddev": pi_stddev, "v": v,
                    "se_v": se_v, "se_pi": se_pi, "r": r}

        if f_create_net is None:
            f_create_net = create_ac_pendulum
        logging.warning("before super(A3CPendulumExp, self).__init__")
        super(A3CPendulumExp, self).__init__(env, f_create_net, episode_n, reward_decay, on_batch_size, off_batch_size,
                                     off_interval, sync_interval, replay_size, prob_min, entropy, l2, optimizer_ctor,
                                     ddqn, aux_r, aux_d)


class A3CPendulum(A3CPendulumExp):
    def __init__(self):
        env = gym.make("Pendulum-v0")
        env = hrl.envs.AugmentEnvWrapper(
            env, reward_decay=0.9, reward_scale=0.1,
            action_limit=np.asarray([env.action_space.low, env.action_space.high])
        )
        super(A3CPendulum, self).__init__(env)
Experiment.register(A3CPendulum, "continuous A3C for Pendulum")


if __name__ == '__main__':
    Experiment.main()
