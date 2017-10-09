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


def wrap_car(env, steer_n, speed_n, frame=4):
    """Apply a common set of wrappers for Atari games."""
    env = CarDiscreteWrapper(env, steer_n, speed_n)
    env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
    # env = ProcessFrame96H(env)
    if frame > 1:
        env = envs.FrameStack(env, frame)
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
            se = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se")

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


class A3CCarContinuous(A3CExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=1000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 1e-4, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = CarGrassWrapper(env, grass_penalty=0.5)
            env = CarContinuousWrapper(env)
            env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
            # env = ProcessFrame96H(env)
            env = envs.FrameStack(env, 4)
            env = envs.ScaledRewards(env, 0.1)
            env = envs.ScaledFloatFrame(env)
        if f_create_net is None:
            dim_action = env.action_space.shape[-1]

            def create_ac_car(inputs):
                l2 = 1e-7
                input_state = inputs[0]
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
                mean = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                   activation_hidden=tf.nn.relu,
                                                   activation_out=None,
                                                   l2=l2,
                                                   var_scope="mean")
                mean = tf.nn.tanh(mean / 4.0)
                stddev = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                     activation_hidden=tf.nn.relu,
                                                     # activation_out=tf.nn.softplus,
                                                     activation_out=None,
                                                     l2=l2,
                                                     var_scope="stddev")
                stddev = 4.0 * tf.nn.sigmoid(stddev / 4.0)
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = create_ac_car
        super(A3CCarContinuous, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                               batch_size)
Experiment.register(A3CCarContinuous, "continuous A3C for CarRacing")


class A3CCarDiscrete2(A3CExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=10000, learning_rate=5e-5, discount_factor=0.99,
                 entropy=hrl.utils.CappedLinear(1e6, 2e-2, 5e-3),
                 batch_size=32):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = wrap_car(env, 3, 3)
        if f_create_net is None:
            dim_action = env.action_space.n

            def create_ac_car(inputs):
                l2 = 1e-7
                input_state = inputs[0]
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
                pi = hrl.utils.Network.layer_fcs(se, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                return {"v": v, "pi": pi}
            f_create_net = create_ac_car
        super(A3CCarDiscrete2, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                              batch_size)
Experiment.register(A3CCarDiscrete2, "continuous A3C for CarRacing")


class DDPGCar(DPGExperiment):
    def __init__(self, env=None, f_net_ddp=None, f_net_dqn=None, episode_n=10000,
                 optimizer_ddp_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-4),
                 optimizer_dqn_ctor=lambda: tf.train.AdamOptimizer(learning_rate=1e-3), target_sync_rate=0.001,
                 ddp_update_interval=4, ddp_sync_interval=4, dqn_update_interval=4, dqn_sync_interval=4,
                 max_gradient=10.0, ou_params=(0.0, 0.15, hrl.utils.CappedLinear(2e5, 1.0, 0.05)), gamma=0.99, batch_size=32, replay_capacity=10000):

        l2 = 1e-8

        def f_actor(input_state, action_shape, is_training):
            se = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se")

            action = hrl.utils.Network.layer_fcs(se, [256], action_shape[0],
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.tanh,
                                                 l2=l2,
                                                 var_scope="action")
            logging.warning("action:%s", action)
            return action

        def f_critic(input_state, input_action, is_training):
            se = hrl.utils.Network.conv2ds(input_state,
                                           shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                           out_flatten=True,
                                           activation=tf.nn.relu,
                                           l2=l2,
                                           var_scope="se")
            se = tf.concat([se, input_action], axis=1)
            q = hrl.utils.Network.layer_fcs(se, [256], 1,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=None,
                                            l2=l2,
                                            var_scope="q")
            q = tf.squeeze(q, axis=1)
            return q
        f_net_dqn = f_critic if f_net_dqn is None else f_net_dqn
        f_net_ddp = f_actor if f_net_ddp is None else f_net_ddp
        if env is None:
            env = gym.make("CarRacing-v0")
            env = CarGrassWrapper(env, grass_penalty=0.5)
            env = CarContinuousWrapper(env)
            env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
            env = envs.FrameStack(env, 4)
            env = envs.ScaledRewards(env, 0.1)
            env = envs.ScaledFloatFrame(env)
            env = envs.AugmentEnvWrapper(env,reward_decay=gamma)

        super(DDPGCar, self).__init__(env, f_net_ddp, f_net_dqn, episode_n, optimizer_ddp_ctor, optimizer_dqn_ctor,
                                      target_sync_rate, ddp_update_interval, ddp_sync_interval, dqn_update_interval,
                                      dqn_sync_interval, max_gradient, ou_params, gamma, batch_size, replay_capacity)

Experiment.register(DDPGCar, "DDPG for CarRacing")


class DQNCarRacing(DQNExperiment):

    def __init__(self, env=None, f_create_q=None, episode_n=10000, discount_factor=0.99, ddqn=False, target_sync_interval=100,
                 target_sync_rate=1.0,
                 update_interval=400,
                 replay_size=2000,
                 batch_size=32,
                 greedy_epsilon=hrl.utils.CappedLinear(1e6, 1.0, 0.05),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        if env is None:
            env = gym.make("CarRacing-v0")
            env = wrap_car(env, 3, 3)
        if f_create_q is None:
            l2=1e-8

            def f_critic(inputs):
                input_state = inputs[0]
                se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se")
                q = hrl.utils.Network.layer_fcs(se, [256], env.action_space.n,
                                                activation_hidden=tf.nn.relu,
                                                activation_out=None,
                                                l2=l2,
                                                var_scope="q")
                return {"q": q}
            f_create_q = f_critic
        super(DQNCarRacing, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                           target_sync_rate, update_interval, replay_size, batch_size, greedy_epsilon,
                                           network_optimizer_ctor)

Experiment.register(DQNCarRacing, "DQN for CarRacing, tuned with ddqn, duel network, etc.")


class I2A(A3CExperimentWithI2A):
    def __init__(self, env=None, f_se = None, f_ac=None, f_env=None, f_rollout=None, f_encoder = None, episode_n=10000,
                 learning_rate=1e-4, discount_factor=0.99, entropy=hrl.utils.CappedLinear(1e6, 1e-1, 1e-4),
                 batch_size=32):
        if env is None:
            env = gym.make('CarRacing-v0')
            env = wrap_car(env, 3, 3, frame=1)
            # env = GoTransposeWrapper(env)
            # env._max_episode_steps = 10000
            # env = envs.FrameStack(env, k=4)
            # env = gym.wrappers.Monitor(env, "./log/AcrobotNew/ICMMaxlen200", force=True)

        if (f_env and f_rollout and f_ac) is None:
            dim_action = env.action_space.n
            dim_observation = env.observation_space.shape

            def create_se(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                se_conv = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="se_conv")
                se_linear = hrl.utils.Network.layer_fcs(se_conv, [], 200,
                                                activation_hidden=tf.nn.relu,
                                                activation_out=tf.nn.relu,
                                                l2=l2,
                                                var_scope="se_linear")
                return {"se": se_linear}

            def create_ac(inputs):
                l2 = 1e-7
                input_feature = inputs[0]

                v = hrl.utils.Network.layer_fcs(input_feature, [256], 1,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="v")
                v = tf.squeeze(v, axis=1)
                pi = hrl.utils.Network.layer_fcs(input_feature, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")

                return {"v": v, "pi": pi}

            def create_rollout(inputs):
                l2 = 1e-7
                input_state = inputs[0]

                # rollout that imitates the A3C policy
                rollout_se = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="rollout_se")

                rollout_action = hrl.utils.Network.layer_fcs(rollout_se, [256], dim_action,
                                                 activation_hidden=tf.nn.relu,
                                                 activation_out=tf.nn.softmax,
                                                 l2=l2,
                                                 var_scope="pi")
                return {"rollout_action": rollout_action}

            def create_env(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_state = tf.squeeze(tf.stack(input_state), axis=0)
                input_action = inputs[1]
                input_action = tf.image.resize_images(tf.reshape(input_action, [-1, 1, 1, dim_action]),
                                                      [dim_observation[0], dim_observation[1]])
                full_input = tf.concat([input_action, input_state], axis=3)

                conv_1 = hrl.utils.Network.conv2ds(full_input,
                                               shape=[(32, 8, 8), (32, 3, 1)],
                                               out_flatten=False,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="conv_1")

                conv_2 = hrl.utils.Network.conv2ds(conv_1,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_2")

                conv_middle = conv_1 + conv_2

                # reward
                conv_r1 = hrl.utils.Network.conv2ds(conv_middle,
                                                   shape=[(32, 3, 1)],
                                                   out_flatten=False,
                                                   activation=tf.nn.relu,
                                                   l2=l2,
                                                   var_scope="conv_r1")

                pool_r1 = tf.layers.max_pooling2d(conv_r1, 2, 1)

                pool_r1 = tf.nn.relu(pool_r1)

                conv_r2 = hrl.utils.Network.conv2ds(pool_r1,
                                                    shape=[(32, 3, 1)],
                                                    out_flatten=False,
                                                    activation=tf.nn.relu,
                                                    l2=l2,
                                                    var_scope="conv_r2")

                pool_r2 = tf.layers.max_pooling2d(conv_r2, 2, 1)

                pool_r2 = tf.nn.relu(pool_r2)

                pool_r2 = tf.contrib.layers.flatten(pool_r2)

                reward = hrl.utils.Network.layer_fcs(pool_r2, [256], 1,
                                                     activation_hidden=tf.nn.relu,
                                                     l2=l2,
                                                     var_scope="reward")
                reward = tf.squeeze(reward, axis=1)

                # next_state
                next_state = hrl.utils.Network.deconv2ds(conv_middle,
                                                         shape=[(3, 8, 8)],
                                                         out_flatten=False,
                                                         activation=tf.nn.relu,
                                                         l2=l2,
                                                         var_scope="next_state")

                return {"next_state": next_state, "reward": reward}

            def create_encoder(inputs):
                l2 = 1e-7
                input_state = inputs[0]
                input_reward = inputs[1]
                print "-------------------------------------"
                print input_state, "\n", input_reward

                rse = hrl.utils.Network.conv2ds(input_state,
                                               shape=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                               out_flatten=True,
                                               activation=tf.nn.relu,
                                               l2=l2,
                                               var_scope="rse")

                re_conv = hrl.utils.Network.layer_fcs(rse, [], 200,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=tf.nn.relu,
                                            l2=l2,
                                            var_scope="re_conv")

                # re_conv = tf.concat([re_conv, tf.reshape(input_reward, [-1, 1])], axis=1)
                re_conv = tf.concat([re_conv, input_reward], axis=1)

                re = hrl.utils.Network.layer_fcs(re_conv, [], 200,
                                            activation_hidden=tf.nn.relu,
                                            activation_out=tf.nn.relu,
                                            l2=l2,
                                            var_scope="re")

                return {"re": re}

            f_se = create_se
            f_ac = create_ac
            f_env = create_env
            f_rollout = create_rollout
            f_encoder = create_encoder

        super(I2A, self).__init__(env, f_se, f_ac, f_env, f_rollout, f_encoder, episode_n, learning_rate,
                                                 discount_factor, entropy, batch_size)


Experiment.register(I2A, "A3C with I2A for complex observation state experiments")


if __name__ == '__main__':
    Experiment.main()
