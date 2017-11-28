import hobotrl as hrl
from hobotrl.experiment import Experiment
import tensorflow as tf
import sys
import gym
import hobotrl.environments as envs
import numpy as np
from environments_recording import EnvRecordingRunner


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


def wrap_car(env, steer_n, speed_n):
    """Apply a common set of wrappers for Atari games."""
    env = CarDiscreteWrapper(env, steer_n, speed_n)
    env = envs.MaxAndSkipEnv(env, skip=2, max_len=1)
    # env = ProcessFrame96H(env)
    env = envs.FrameStack(env, 4)
    env = envs.ScaledRewards(env, 0.1)
    env = envs.ScaledFloatFrame(env)
    return env


class A3CRecordingExperiment(Experiment):
    def __init__(self,
                 env, f_create_net, episode_n=1000,
                 learning_rate=1e-4,
                 discount_factor=0.9,
                 entropy=1e-2,
                 batch_size=8
                 ):
        super(A3CRecordingExperiment, self).__init__()
        self._env, self._f_create_net, self._episode_n, self._learning_rate, \
            self._discount_factor, self._entropy, self._batch_size = \
            env, f_create_net, episode_n, learning_rate, \
            discount_factor, entropy, batch_size

    def run(self, args):
        state_shape = list(self._env.observation_space.shape)

        def create_optimizer():
            return tf.train.AdamOptimizer(self._learning_rate)

        def create_agent(n_optimizer, global_step):
            # all ScheduledParam hyper parameters are mutable objects.
            # so we will not want to use same object for different Agent instances.
            entropy = hrl.utils.clone_params(self._entropy)
            agent = hrl.ActorCritic(
                f_create_net=self._f_create_net,
                state_shape=state_shape,
                # ACUpdate arguments
                discount_factor=self._discount_factor,
                entropy=entropy,
                target_estimator=None,
                max_advantage=100.0,
                # optimizer arguments
                network_optimizer=n_optimizer,
                # sampler arguments
                sampler=None,
                batch_size=self._batch_size,
                global_step=global_step,
            )
            return agent

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        agent = hrl.async.ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.create_session(config=config) as sess:
            agent.set_session(sess)
            runner = EnvRecordingRunner(self._env, agent, reward_decay=self._discount_factor,  max_episode_len=1000,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        render_once=True,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(self._episode_n)


class A3CCarRecordingDiscrete2(A3CRecordingExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=100, learning_rate=5e-5, discount_factor=0.99,
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
        super(A3CCarRecordingDiscrete2, self).__init__(env, f_create_net, episode_n, learning_rate, discount_factor, entropy,
                                              batch_size)

Experiment.register(A3CCarRecordingDiscrete2, "Discrete A3C for CarRacing Recording")


# class ACRecordingExperiment(Experiment):
#     def __init__(self,
#                  env, f_create_net, episode_n=1000,
#                  learning_rate=1e-4,
#                  discount_factor=0.9,
#                  entropy=1e-2,
#                  batch_size=8
#                  ):
#         super(A3CRecordingExperiment, self).__init__()
#         self._env, self._f_create_net, self._episode_n, self._learning_rate, \
#             self._discount_factor, self._entropy, self._batch_size = \
#             env, f_create_net, episode_n, learning_rate, \
#             discount_factor, entropy, batch_size
#
#     def run(self, args):
#         state_shape = list(self._env.observation_space.shape)
#         global_step = tf.get_variable(
#             'global_step', [], dtype=tf.int32,
#             initializer=tf.constant_initializer(0), trainable=False
#         )
#         entropy = hrl.utils.clone_params(self._entropy)
#         agent = hrl.ActorCritic(
#             f_create_net=self._f_create_net,
#             state_shape=state_shape,
#             # ACUpdate arguments
#             discount_factor=self._discount_factor,
#             entropy=entropy,
#             target_estimator=None,
#             max_advantage=100.0,
#             # optimizer arguments
#             network_optimizer=tf.train.AdamOptimizer(self._learning_rate),
#             # sampler arguments
#             sampler=None,
#             batch_size=self._batch_size,
#             global_step=global_step,
#         )
#
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#
#         restore_var_list = []
#         for var in tf.global_variables():
#             print "var_name: ", var.name
#             if 'Adam' in var.name or 'optimizers/beta1_power' in var.name \
#                     or 'optimizers/beta2_power' in var.name \
#                     or var.name == 'global_step:0':
#                 pass
#             else:
#                 restore_var_list.append(var)
#
#         with agent.create_session(config=config, save_dir=args.logdir, save_checkpoint_secs=1200,
#                               restore_var_list=restore_var_list) as sess:
#             agent.set_session(sess)
#             runner = EnvRecordingRunner(self._env, agent, reward_decay=self._discount_factor,  max_episode_len=1000,
#                                         evaluate_interval=sys.maxint, render_interval=args.render_interval,
#                                         render_once=True,
#                                         logdir=args.logdir if args.index == 0 else None,
#                                         savedir=args.savedir+"/"+args.index)
#             runner.episode(self._episode_n)
#
