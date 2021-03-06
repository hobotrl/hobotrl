# -*- coding: utf-8 -*-

import sys

from hobotrl.experiment import ParallelGridSearch
from hobotrl.policy import OUNoise, OUNoise2

sys.path.append(".")
import gym.spaces

import hobotrl as hrl
from hobotrl.utils import CappedLinear
from exp_algorithms import *
import hobotrl.algorithms.ot as ot
import exp_algorithms as alg


class ACPendulum(ACExperiment):

    def __init__(self, env=None, f_create_net=None, episode_n=2000, discount_factor=0.9, entropy=3e-2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), batch_size=8):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.C2DEnvWrapper(env, [5])
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        if f_create_net is None:
            def f_net(inputs):
                l2 = 1e-4
                state = inputs[0]
                q = hrl.network.Utils.layer_fcs(state, [200, 100], env.action_space.n, l2=l2, var_scope="q")
                pi = hrl.network.Utils.layer_fcs(state, [200, 100], env.action_space.n, activation_out=tf.nn.softmax,
                                                 l2=l2, var_scope="pi")
                return {"q": q, "pi": pi}
            f_create_net = f_net
        super(ACPendulum, self).__init__(env, f_create_net, episode_n, discount_factor, entropy,
                                         network_optimizer_ctor, batch_size)
Experiment.register(ACPendulum, "discrete actor critic for Pendulum")


class ACOOPendulum(ACOOExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=1000, reward_decay=0.9, on_batch_size=8, off_batch_size=32,
                 off_interval=0, sync_interval=1000, replay_size=10000, prob_min=5e-3,
                 entropy=hrl.utils.CappedLinear(4e5, 1e-2, 1e-3), l2=1e-8,
                 optimizer_ctor=lambda: tf.train.AdamOptimizer(1e-3), ddqn=False, aux_r=False, aux_d=False):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.C2DEnvWrapper(env, [5])
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.1)
        if f_create_net is None:
            def f_net(inputs):
                input_state = inputs[0]
                se = hrl.network.Utils.layer_fcs(input_state,
                                                 shape=[200],
                                                 out_count=100,
                                                 activation_hidden=tf.nn.elu,
                                                 activation_out=tf.nn.elu,
                                                 l2=l2,
                                                 var_scope="se")

                q = hrl.utils.Network.layer_fcs(se, [256], env.action_space.n,
                                                activation_hidden=tf.nn.relu,
                                                l2=l2,
                                                var_scope="q")
                pi = hrl.utils.Network.layer_fcs(se, [256], env.action_space.n,
                                                 activation_hidden=tf.nn.relu,
                                                 # activation_out=tf.nn.softplus,
                                                 l2=l2,
                                                 var_scope="pi")
                pi = tf.nn.softmax(pi)
                # pi = pi + prob_min
                # pi = pi / tf.reduce_sum(pi, axis=-1, keep_dims=True)
                return {"pi": pi, "q": q, "se": se}
            f_create_net = f_net
        super(ACOOPendulum, self).__init__(env, f_create_net, episode_n, reward_decay, on_batch_size, off_batch_size,
                                           off_interval, sync_interval, replay_size, prob_min, entropy, l2,
                                           optimizer_ctor, ddqn, aux_r, aux_d)
Experiment.register(ACOOPendulum, "discrete actor critic for Pendulum")


class ACContinuousPendulum(ACExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=2000, discount_factor=0.9,
                 # entropy=CappedLinear(1e5, 1e-1, 1e-2),
                 entropy=3e-2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), batch_size=16):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        dim_action = env.action_space.shape[-1]
        if f_create_net is None:
            def f_net_sigmoid(inputs):
                l2 = 1e-8
                state = inputs[0]
                v = hrl.network.Utils.layer_fcs(state, [200, 100], 1, l2=l2, var_scope="v")
                v = tf.squeeze(v, axis=1)
                mean = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                   activation_hidden=tf.nn.elu,
                                                   activation_out=tf.tanh,
                                                 l2=l2, var_scope="mean")
                stddev = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                     activation_hidden=tf.nn.elu,
                                                     activation_out=lambda x: 2.0 * tf.sigmoid(x / 2.0),
                                                     l2=l2, var_scope="stddev")
                stddev = 0.1 * stddev + tf.stop_gradient(0.9 * stddev)
                return {"v": v, "mean": mean, "stddev": stddev}

            def f_net(inputs):
                l2 = 1e-8
                state = inputs[0]
                v = hrl.network.Utils.layer_fcs(state, [200, 100], 1, l2=l2, var_scope="v")
                v = tf.squeeze(v, axis=1)
                mean = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action, activation_out=tf.tanh,
                                                   l2=l2, var_scope="mean")
                stddev = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                     activation_out=lambda x: tf.nn.softplus(x/4.0),
                                                     l2=l2, var_scope="stddev")
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = f_net_sigmoid

        super(ACContinuousPendulum, self).__init__(env, f_create_net, episode_n, discount_factor, entropy,
                                                   network_optimizer_ctor, batch_size)
Experiment.register(ACContinuousPendulum, "continuous actor critic for Pendulum")


class ACConPendulumSearch(hrl.experiment.GridSearch):
    def __init__(self):
        super(ACConPendulumSearch, self).__init__(ACContinuousPendulum, {
            "entropy": [CappedLinear(1e4, 1e-1, 1e-2), 1e-1],
            "batch_size": [16],
            "episode_n": [5],
        })
Experiment.register(ACConPendulumSearch, "continuous actor critic for Pendulum")


class DQNPendulum(DQNExperiment):

    def __init__(self, env=None, f_create_q=None, episode_n=200, discount_factor=0.99, ddqn=False, target_sync_interval=100,
                 target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=32, greedy_epsilon=0.1,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.C2DEnvWrapper(env, [5])
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1)
        if f_create_q is None:
            l2 = 1e-10

            def f_net(inputs):
                input_state = inputs[0]
                fc_out = hrl.utils.Network.layer_fcs(
                    input_state, [200, 200], env.action_space.n,
                    activation_hidden=tf.nn.relu, activation_out=None, l2=l2
                )
                return {"q": fc_out}
            f_create_q = f_net
        super(DQNPendulum, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                          target_sync_rate, update_interval, replay_size, batch_size, greedy_epsilon,
                                          network_optimizer_ctor)

Experiment.register(DQNPendulum, "DQN for Pendulum")


class DDQNPendulum(DQNPendulum):

    def __init__(self, env=None, f_create_q=None, episode_n=1000, discount_factor=0.99, ddqn=True,
                 target_sync_interval=100, target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=32,
                 greedy_epsilon=0.3,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        super(DDQNPendulum, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                           target_sync_rate, update_interval, replay_size, batch_size, greedy_epsilon,
                                           network_optimizer_ctor)
Experiment.register(DDQNPendulum, "Double DQN for Pendulum")


class DuelDQNPendulum(DQNPendulum):

    def __init__(self, env=None, f_create_q=None, episode_n=1000, discount_factor=0.99, ddqn=False,
                 target_sync_interval=100, target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=32,
                 greedy_epsilon=0.3,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        if f_create_q is None:
            def f_net(inputs, num_action, is_training):
                input_var = inputs
                se = hrl.utils.Network.layer_fcs(
                    input_var, [200, 200], num_action,
                    activation_hidden=tf.nn.relu, activation_out=tf.nn.relu, l2=1e-4
                )
                v = hrl.utils.Network.layer_fcs(se, [100], 1, var_scope="v")
                a = hrl.utils.Network.layer_fcs(se, [100], num_action, var_scope="a")
                a = a - tf.reduce_mean(a, axis=1, keep_dims=True)
                q = a + v
                return {"q": q}
            f_create_q = f_net

        super(DuelDQNPendulum, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                              target_sync_rate, update_interval, replay_size, batch_size,
                                              greedy_epsilon, network_optimizer_ctor)
Experiment.register(DuelDQNPendulum, "Duel DQN for Pendulum")


class DPGPendulum(DPGExperiment):
    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, episode_n=100, discount_factor=0.9,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0),
                 ou_params=(0, 0.2, hrl.utils.CappedLinear(1e5, 0.5, 0.1)),
                 target_sync_interval=10, target_sync_rate=0.01, batch_size=32, replay_capacity=1000, **kwargs):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        state_shape = list(env.observation_space.shape)
        dim_action = env.action_space.shape[-1]
        l2 = 1e-8
        if f_se is None:
            def f(inputs):
                return {"se": inputs[0]}
            f_se = f
        if f_actor is None:
            def f(inputs):
                se = inputs[0]
                actor = hrl.network.Utils.layer_fcs(se, [256, 256], dim_action, activation_out=tf.nn.tanh, l2=l2,
                                                    var_scope="action")
                return {"action": actor}
            f_actor = f
        if f_critic is None:
            def f(inputs):
                se, action = inputs[0], inputs[1]
                se = tf.concat([se, action], axis=-1)
                q = hrl.network.Utils.layer_fcs(se, [256, 256], 1, activation_out=None, l2=l2, var_scope="q")
                q = tf.squeeze(q, axis=1)
                return {"q": q}
            f_critic = f

        super(DPGPendulum, self).__init__(env, f_se, f_actor, f_critic, episode_n, discount_factor,
                                          network_optimizer_ctor, ou_params, target_sync_interval, target_sync_rate,
                                          batch_size, replay_capacity, **kwargs)
Experiment.register(DPGPendulum, "DPG for Pendulum")


class DPGBipedal(DPGPendulum):

    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None,
                 episode_n=2000, discount_factor=0.9,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0),
                 ou_params=(0, 0.2, hrl.utils.CappedExp(2e5, 0.5, 0.02)),
                 target_sync_interval=1,
                 target_sync_rate=0.001,
                 batch_size=128,
                 state_skip=4,
                 reward_scale=0.5,
                 replay_capacity=100000, **kwargs):
        if env is None:
            env = gym.make("BipedalWalker-v2")
            env = MaxAndSkipEnv(env, max_len=1, skip=state_skip)
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=reward_scale)

        super(DPGBipedal, self).__init__(env, f_se, f_actor, f_critic, episode_n, discount_factor,
                                         network_optimizer_ctor, ou_params, target_sync_interval, target_sync_rate,
                                         batch_size, replay_capacity, **kwargs)
Experiment.register(DPGBipedal, "DPG for Bipedal")


class DPGBipedal2(DPGBipedal):

    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, episode_n=2000, discount_factor=0.9,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0),
                 noise_type=OUNoise2,
                 ou_params=(0, hrl.utils.Cosine(1e4, 0.98, 0.8), hrl.utils.CappedExp(2e5, 2.0, 0.2)),
                 target_sync_interval=1,
                 target_sync_rate=0.001,
                 batch_size=128,
                 state_skip=1,
                 reward_scale=0.5,
                 replay_capacity=100000):
        super(DPGBipedal2, self).__init__(env, f_se, f_actor, f_critic, episode_n, discount_factor,
                                          network_optimizer_ctor, ou_params, target_sync_interval, target_sync_rate,
                                          batch_size, state_skip, reward_scale, replay_capacity)
Experiment.register(DPGBipedal2, "DPG for Bipedal, test for new noise")


class DPGBipedalSearch(ParallelGridSearch):

    def __init__(self):
        super(DPGBipedalSearch, self).__init__(DPGBipedal, [
            # {
            #     "episode_n": [1200],
            #     "replay_capacity": [100000, 10000, 1000],
            #     "batch_size": [128],
            #     "state_skip": [4],
            #     "ou_params": [(0, 0.2, hrl.utils.CappedExp(2e5, 0.5, 0.02))]
            # },
            # {
            #     "episode_n": [1200],
            #     "replay_capacity": [100000],
            #     "batch_size": [32],
            #     "state_skip": [4],
            #     "ou_params": [(0, 0.2, hrl.utils.CappedExp(2e5, 0.5, 0.02))]
            # },
            # {
            #     "episode_n": [1200],
            #     "replay_capacity": [100000],
            #     "batch_size": [128],
            #     "state_skip": [2, 1],
            #     "ou_params": [(0, 0.2, hrl.utils.CappedExp(2e5, 0.5, 0.02))]
            # },
            {
                "episode_n": [1200],
                "replay_capacity": [100000],
                "batch_size": [128],
                "state_skip": [4],
                "noise_type": [OUNoise2],
                "ou_params": [(0, 0.8, hrl.utils.CappedExp(2e5, 2.0, 0.2)),
                              (0, 0.8, hrl.utils.CappedExp(2e5, 1.5, 0.2)),
                              (0, 0.8, hrl.utils.CappedExp(2e5, 1.5, 0.05)),
                              (0, 0.8, hrl.utils.CappedExp(2e5, 2.0, 0.05)),
                              ]
            },
        ], parallel=4)
Experiment.register(DPGBipedalSearch, "Search for DPG for Bipedal")


class CarEnvWrapper(object):
    """
    Wraps car env into discrete action control problem
    """

    def __init__(self, env, steer_n, speed_n):
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

    def step(self, *args, **kwargs):
        # lives_before = self.env.ale.lives()
        if len(args) > 0:
            action_i = args[0]
        else:
            action_i = kwargs["action"]
        action_c = self.action_d2c(action_i)
        # logging.warning("action d2c: %s => %s", action_i, action_c)
        next_state, reward, done, info = self.env.step(action_c)
        # lives_after = self.env.ale.lives()
        #
        # # End the episode when a life is lost
        # if lives_before > lives_after:
        #   done = True
        #
        # # Clip rewards to [-1,1]
        # reward = max(min(reward, 1), -1)

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


class PERDQNPendulum(alg.PERDQNExperiment):

    def __init__(self, env=None, f_create_q=None, episode_n=1000, discount_factor=0.9, ddqn=False,
                 target_sync_interval=10,
                 target_sync_rate=1.0, update_interval=1, replay_size=1000, batch_size=8,
                 priority_bias=0.5,
                 importance_weight=CappedLinear(2e5, 0.5, 1.0),
                 greedy_epsilon=0.2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.C2DEnvWrapper(env, [5])
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=0.9, reward_scale=0.1)

        if f_create_q is None:
            def f_net(inputs):
                input_var = inputs[0]
                fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], env.action_space.n,
                                                     activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4,
                                                     var_scope="q")
                return {"q": fc_out}
            f_create_q = f_net

        super(PERDQNPendulum, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                             target_sync_rate, update_interval, replay_size, batch_size, priority_bias,
                                             importance_weight, greedy_epsilon, network_optimizer_ctor)

Experiment.register(PERDQNPendulum, "Prioritized Exp Replay with DQN, for Pendulum")


class OTDQNPendulum(OTDQNExperiment):

    """
    converges on Pendulum.
    However, in Pendulum, weight_upper > 0 hurts performance.
    should verify on more difficult problems
    """

    def __init__(self, env=None, f_create_q=None, episode_n=1000,
                 discount_factor=0.99, ddqn=False, target_sync_interval=100,
                 target_sync_rate=1.0, update_interval=4, replay_size=1000, batch_size=8, lower_weight=1.0,
                 upper_weight=1.0, neighbour_size=8, greedy_epsilon=0.2,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0)):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.C2DEnvWrapper(env, [5])
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        if f_create_q is None:
            def f_net(inputs):
                input_state = inputs[0]
                fc_out = hrl.utils.Network.layer_fcs(input_state, [200, 200], env.action_space.n,
                                                     activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
                return {"q": fc_out}
            f_create_q = f_net
        super(OTDQNPendulum, self).__init__(env, f_create_q, episode_n, discount_factor, ddqn, target_sync_interval,
                                            target_sync_rate, update_interval, replay_size, batch_size, lower_weight,
                                            upper_weight, neighbour_size, greedy_epsilon, network_optimizer_ctor)
    #
    # def run(self, args):
    #     discount_factor = 0.9
    #     K = 8
    #     batch_size = 4
    #     weight_lower = 0.0
    #     weight_upper = 0.0
    #     target_sync_interval = 10
    #     target_sync_rate = 0.01
    #     replay_size = 1000
    #
    #     env = gym.make("Pendulum-v0")
    #     env = hrl.envs.C2DEnvWrapper(env, [5])
    #     env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
    #
    #     optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #
    #     training_params = (optimizer_td, target_sync_rate, 10.0)
    #
    #     def f_net(inputs):
    #         input_state = inputs[0]
    #         fc_out = hrl.utils.Network.layer_fcs(input_state, [200, 200], env.action_space.n,
    #                                              activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
    #         return {"q": fc_out}
    #
    #     state_shape = list(env.observation_space.shape)
    #     global_step = tf.get_variable('global_step', [],
    #                                   dtype=tf.int32,
    #                                   initializer=tf.constant_initializer(0),
    #                                   trainable=False)
    #     agent = ot.OTDQN(
    #         f_create_q=f_net,
    #         lower_weight=weight_lower, upper_weight=weight_upper, neighbour_size=K,
    #         state_shape=state_shape, num_actions=env.action_space.n, discount_factor=discount_factor,
    #         target_sync_interval=target_sync_interval, target_sync_rate=target_sync_rate,
    #         greedy_epsilon=0.2,
    #         network_optimizer=None, max_gradient=10.0,
    #         update_interval=2,
    #         replay_size=replay_size, batch_size=batch_size, sampler=None,
    #         global_step=global_step
    #     )
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     with agent.create_session(config=config, save_dir=args.logdir) as sess:
    #         agent.set_session(sess)
    #         runner = hrl.envs.EnvRunner(env, agent, reward_decay=discount_factor,
    #                                     evaluate_interval=sys.maxint, render_interval=args.render_interval,
    #                                     logdir=args.logdir)
    #         runner.episode(500)

Experiment.register(OTDQNPendulum, "Optimaly Tightening DQN for Pendulum")


class AOTDQNPendulum(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    should verify on more difficult problems
    """
    def run(self, args):
        reward_decay = 0.9
        K = 4
        batch_size = 8
        weight_lower = 1.0
        weight_upper = 1.0
        replay_size = 1000
        training_params = (tf.train.AdamOptimizer(learning_rate=0.001), 0.01, 10.0)
        env = gym.make("Pendulum-v0")
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.1)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            fc_out = hrl.utils.Network.layer_fcs(input_var, [200, 200], num_action,
                                                 activation_hidden=tf.nn.relu, activation_out=None, l2=1e-4)
            return fc_out

        state_shape = list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        agent = ot.OTDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # OTDQN
            f_net_dqn=f_net,
            state_shape=state_shape,
            num_actions=env.action_space.n,
            reward_decay=reward_decay,
            batch_size=batch_size,
            K=K,
            weight_lower_bound=weight_lower,
            weight_upper_bound=weight_upper,
            training_params=training_params,
            schedule=(1, 10),
            replay_capacity=replay_size,
            # BaseDeepAgent
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        logdir=args.logdir)
            runner.episode(500)

Experiment.register(AOTDQNPendulum, "Optimaly Tightening DQN for Pendulum")


class AOTDQNBreakout(Experiment):
    """
    converges on Pendulum. OT implementation from hobotrl.algorithms package.
    However, in Pendulum, weight_upper > 0 hurts performance.
    should verify on more difficult problems
    """
    def run(self, args):
        reward_decay = 0.9
        K = 4
        batch_size = 8
        weight_lower = 1.0
        weight_upper = 1.0
        replay_size = 1000

        training_params = (tf.train.AdamOptimizer(learning_rate=1e-4), 0.01, 10.0)
        env = gym.make("Breakout-v0")
        # env = hrl.envs.C2DEnvWrapper(env, [5])

        def state_trans(state):
            gray = np.asarray(np.dot(state, [0.299, 0.587, 0.114]))
            gray = cv2.resize(gray, (84, 84))
            return np.asarray(gray.reshape(gray.shape + (1,)), dtype=np.uint8)

        env = hrl.envs.AugmentEnvWrapper(env, reward_decay=reward_decay, reward_scale=0.1,
                                         state_augment_proc=state_trans, state_stack_n=4)

        def f_net(inputs, num_action, is_training):
            input_var = inputs
            print "input size:", input_var
            out = hrl.utils.Network.conv2d(input_var=input_var, h=8, w=8, out_channel=32,
                                           strides=[4, 4], activation=tf.nn.relu, var_scope="conv1")
            # 20 * 20 * 32
            print "out size:", out
            out = hrl.utils.Network.conv2d(input_var=out, h=4, w=4, out_channel=64,
                                           strides=[2, 2], activation=tf.nn.relu, var_scope="conv2")
            # 9 * 9 * 64
            print "out size:", out
            out = hrl.utils.Network.conv2d(input_var=out, h=3, w=3, out_channel=64,
                                           strides=[1, 1], activation=tf.nn.relu, var_scope="conv3")
            # 7 * 7 * 64
            print "out size:", out
            out = tf.reshape(out, [-1, 7 * 7 * 64])
            out = hrl.utils.Network.layer_fcs(input_var=out, shape=[512], out_count=num_action,
                                              activation_hidden=tf.nn.relu,
                                              activation_out=None, var_scope="fc")
            return out

        state_shape = [84, 84, 4]  # list(env.observation_space.shape)
        global_step = tf.get_variable('global_step', [],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        agent = ot.OTDQN(
            # EpsilonGreedyPolicyMixin params
            actions=range(env.action_space.n),
            epsilon=0.2,
            # OTDQN
            f_net_dqn=f_net,
            state_shape=state_shape,
            num_actions=env.action_space.n,
            reward_decay=reward_decay,
            batch_size=batch_size,
            K=K,
            weight_lower_bound=weight_lower,
            weight_upper_bound=weight_upper,
            training_params=training_params,
            schedule=(1, 10),
            replay_capacity=replay_size,
            state_offset_scale=(-128, 1.0 / 128),
            # BaseDeepAgent
            global_step=global_step
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with agent.create_session(config=config, save_dir=args.logdir) as sess:
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=reward_decay,
                                        evaluate_interval=sys.maxint, render_interval=args.render_interval,
                                        logdir=args.logdir)
            runner.episode(500)

Experiment.register(AOTDQNBreakout, "Optimaly Tightening DQN for Breakout")


class BootstrappedDQNSnakeGame(Experiment):
    def run(self, args):
        """
        Run the experiment.
        """
        def render():
            """
            Render the environment and related information to the console.
            """
            if not display:
                return

            print env.render(mode='ansi')
            print "Reward:", reward
            print "Head:", agent.current_head
            print "Done:", done
            print ""
            time.sleep(frame_time)

        from hobotrl.environments import SnakeGame
        from hobotrl.algorithms.bootstrapped_DQN import BootstrappedDQN
        from hobotrl.environments import EnvRunner2

        import time
        import os
        import random

        # Parameters
        random.seed(1105)  # Seed

        for n_head in [1, 3, 5, 10, 15, 20, 30]:

            log_dir = os.path.join(args.logdir, "head%d" % n_head)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file_name = "booststrapped_DQN_Snake.csv"

            # Initialize the environment and the agent
            env = SnakeGame(3, 3, 1, 1, max_episode_length=50)
            agent = BootstrappedDQN(observation_space=env.observation_space,
                                    action_space=env.action_space,
                                    reward_decay=1.,
                                    td_learning_rate=0.5,
                                    target_sync_interval=2000,
                                    nn_constructor=self.nn_constructor,
                                    loss_function=self.loss_function,
                                    trainer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize,
                                    replay_buffer_class=hrl.playback.MapPlayback,
                                    replay_buffer_args={"capacity": 10000},
                                    min_buffer_size=100,
                                    batch_size=8,
                                    n_heads=n_head)

            # Start training
            env_runner = EnvRunner2(env=env,
                                    agent=agent,
                                    n_episodes=3000,
                                    moving_average_window_size=100,
                                    no_reward_reset_interval=-1,
                                    checkpoint_save_interval=100000,
                                    log_dir=log_dir,
                                    log_file_name=log_file_name,
                                    render_env=False,
                                    render_interval=1000,
                                    render_length=200,
                                    frame_time=0.1,
                                    render_options={"mode": "ansi"}
                                    )
            env_runner.run()

    @staticmethod
    def loss_function(output, target):
        """
        Calculate the loss.
        """
        return tf.reduce_sum(tf.sqrt(tf.squared_difference(output, target)+1)-1, axis=-1)

    @staticmethod
    def nn_constructor(observation_space, action_space, n_heads, **kwargs):
        """
        Construct the neural network.
        """
        def leakyRelu(x):
            return tf.maximum(0.01*x, x)

        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

        def weight(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        def bias(shape):
            return tf.Variable(tf.constant(0.1, shape=shape))

        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        eshape = observation_space.shape
        nn_outputs = []

        # Layer 1 parameters
        n_channel1 = 8
        w1 = weight([3, 3, eshape[-1], n_channel1])
        b1 = bias([n_channel1])

        # Layer 2 parameters
        n_channel2 = 16
        w2 = weight([n_channel1*eshape[0]*eshape[1], n_channel2])
        b2 = bias([n_channel2])

        # Layer 1
        layer1 = leakyRelu(conv2d(x, w1) + b1)
        layer1_flatten = tf.reshape(layer1, [-1, n_channel1*eshape[0]*eshape[1]])

        # Layer 2
        layer2 = leakyRelu(tf.matmul(layer1_flatten, w2) + b2)

        for i in range(n_heads):
            # Layer 3 parameters
            w3 = weight([n_channel2, 4])
            b3 = bias([4])

            # Layer 3
            layer3 = tf.matmul(layer2, w3) + b3

            nn_outputs.append(layer3)

        return {"input": x, "head": nn_outputs}

Experiment.register(BootstrappedDQNSnakeGame, "Bootstrapped DQN for the Snake game")


from hobotrl.algorithms.bootstrapped_DQN import BootstrappedDQN


class BootstrappedDQNAtari(Experiment):
    def __init__(self, env, augment_wrapper_args={}, agent_args={}, runner_args={},
                 stack_n=4, frame_skip_n=4, reward_decay=0.99,
                 agent_type=BootstrappedDQN):
        """
        Base class Experiments in Atari games.

        :param env: environment.
        :param augment_wrapper_args(dict): arguments for "AugmentEnvWrapper".
        :param agent_args(dict): arguments for the agent.
        :param runner_args(dict): arguments for the environment runner.
        :param stack_n(int): number of frames to stack in total.
        :param frame_skip_n(int): number of frames to skip.
        :param agent_type(class): class name of the agent.
        """
        assert stack_n >= 1
        assert 1 <= frame_skip_n <= stack_n
        assert stack_n % frame_skip_n == 0
        assert 0. <= reward_decay <= 1.

        import math

        Experiment.__init__(self)

        n_head = 10  # Number of heads

        self.augment_wrapper_args = augment_wrapper_args
        self.agent_args = agent_args
        self.runner_args = runner_args

        # Wrap the environment
        history_stack_n = stack_n//frame_skip_n
        augment_wrapper_args = {"reward_decay": math.pow(reward_decay, 1.0/history_stack_n),
                                "reward_scale": 1.,
                                "state_augment_proc": self.state_trans,
                                "state_skip": frame_skip_n,
                                "state_scale": 1.0/255.0,
                                "discard_skipped_frames": False,
                                "random_start": True}
        augment_wrapper_args.update(self.augment_wrapper_args)
        env = self.env = hrl.envs.AugmentEnvWrapper(env, **augment_wrapper_args)
        env = self.env = hrl.envs.StateHistoryStackEnvWrapper(env,
                                                              stack_n=history_stack_n)

        # Initialize the agent
        agent_args = {"reward_decay": math.pow(reward_decay, 1.0/history_stack_n),
                      "td_learning_rate": 1.,
                      "target_sync_interval": 1000,
                      "nn_constructor": self.nn_constructor,
                      "loss_function": self.loss_function,
                      "trainer": tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize,
                      "replay_buffer_class": hrl.playback.MapPlayback,
                      "replay_buffer_args": {"capacity": 10000},
                      "min_buffer_size": 5000,
                      "batch_size": 8,
                      "n_heads": n_head}
        agent_args.update(self.agent_args)
        self.agent = agent_type(observation_space=env.observation_space,
                                action_space=env.action_space,
                                **agent_args)

    @staticmethod
    def state_trans(state):
        """
        Transform the state to 84*84 grayscale image.

        :param state: state.
        :return: transformed image.
        """
        gray = np.asarray(np.dot(state, [0.299, 0.587, 0.114]))
        gray = cv2.resize(gray, (84, 84))

        return np.asarray(gray.reshape(gray.shape + (1,)), dtype=np.int8)

    @staticmethod
    def show_state_trans_result_wrapper(state):
        """
        Transform the state with "state_trans" and show the result in the image viewer.
        Used to visualize the result of state_trans and should not be used during training.

        :param state: state.
        :return: transformed image returned by state_trans.
        """
        global image_viewer
        import gym.envs.classic_control.rendering as rendering

        # Initialize image viewer if needed
        try:
            image_viewer
        except NameError:
            image_viewer = rendering.SimpleImageViewer()

        # Transform with state_trans
        image = BootstrappedDQNAtari.state_trans(state)

        # Resize the image to see it clearly
        im_view = image.reshape((84, 84))
        im_view = np.array(im_view, dtype=np.float32)
        im_view = cv2.resize(im_view, (336, 336), interpolation=cv2.INTER_NEAREST)
        im_view = np.array(im_view, dtype=np.int8)
        im_view = np.stack([im_view]*3, axis=-1)

        # Show image
        image_viewer.imshow(im_view)
        return image

    def run(self, args, checkpoint_number=None):
        """
        Run the experiment.

        :param args: arguments.
        :param checkpoint_number: if not None, checkpoint will be loaded before training.
        """
        from hobotrl.environments import EnvRunner2
        import os

        # Create logging folder if needed
        log_dir = args.logdir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = "booststrapped_DQN.csv"

        # Initialize the environment runner
        runner_args = {"n_episodes": -1,
                       "moving_average_window_size": 100,
                       "no_reward_reset_interval": -1,
                       "checkpoint_save_interval": 100000,
                       "render_env": False,
                       "show_frame_rate": True,
                       "show_frame_rate_interval": 2000}
        runner_args.update(self.runner_args)
        env_runner = EnvRunner2(env=self.env,
                                agent=self.agent,
                                log_dir=log_dir,
                                log_file_name=log_file_name,
                                **runner_args)

        # Load checkpoint if needed
        if checkpoint_number:
            checkpoint_file_name = '%d.ckpt' % checkpoint_number
            env_runner.load_checkpoint(checkpoint_file_name, checkpoint_number)

        # Start training
        env_runner.run()

    @staticmethod
    def loss_function(output, target):
        """
        Calculate the loss.
        """
        return tf.reduce_sum(tf.sqrt(tf.squared_difference(output, target)+1)-1, -1)

    @staticmethod
    def nn_constructor(observation_space, action_space, n_heads, **kwargs):
        """
        Construct the neural network.
        """
        def leakyRelu(x):
            return tf.maximum(0.01*x, x)

        import tensorflow.contrib.layers as layers
        nn_outputs = []

        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        print "input size:", x
        out = hrl.utils.Network.conv2d(input_var=x, h=8, w=8, out_channel=32,
                                       strides=[4, 4], activation=leakyRelu, padding="VALID", var_scope="conv1")
        # 20 * 20 * 32
        print "out size:", out
        out = hrl.utils.Network.conv2d(input_var=out, h=4, w=4, out_channel=64,
                                       strides=[2, 2], activation=leakyRelu, padding="VALID", var_scope="conv2")
        # 9 * 9 * 64
        print "out size:", out
        out = hrl.utils.Network.conv2d(input_var=out, h=3, w=3, out_channel=64,
                                       strides=[1, 1], activation=leakyRelu, padding="VALID", var_scope="conv3")

        # 7 * 7 * 64
        out = tf.reshape(out, [-1, int(np.product(out.shape[1:]))])
        out = layers.fully_connected(out, 512, activation_fn=leakyRelu)
        print "out size:", out

        for _ in range(n_heads):
            head = layers.fully_connected(out, action_space.n, activation_fn=None)

            nn_outputs.append(head)

        return {"input": x, "head": nn_outputs}


class BootstrappedDQNBattleZone(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('BattleZone-v0'),
                                      augment_wrapper_args={"reward_scale": 0.001},
                                      agent_args={"replay_buffer_args": {"capacity": 10000},
                                                  "min_buffer_size": 10000})

Experiment.register(BootstrappedDQNBattleZone, "Bootstrapped DQN for the BattleZone")


class BootstrappedDQNBreakOut(BootstrappedDQNAtari):
    def __init__(self):
        from hobotrl.algorithms.bootstrapped_DQN import bernoulli_mask
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('Breakout-v0'),
                                      runner_args={"no_reward_reset_interval": 2000},
                                      agent_args={"n_heads": 30,
                                                  "bootstrap_mask": bernoulli_mask(0.2)}
                                      )

Experiment.register(BootstrappedDQNBreakOut, "Bootstrapped DQN for the BreakOut")


class BootstrappedDQNPong(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self, gym.make('PongNoFrameskip-v4'))

Experiment.register(BootstrappedDQNPong, "Bootstrapped DQN for the Pong")


class BootstrappedDQNEnduro(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('Enduro-v0'),
                                      augment_wrapper_args={
                                          "reward_scale": 0.3
                                          },
                                      agent_args={
                                          "batch_size": 8
                                      })

Experiment.register(BootstrappedDQNEnduro, "Bootstrapped DQN for the Enduro")


class BootstrappedDQNIceHockey(BootstrappedDQNAtari):
    def __init__(self):
        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('IceHockey-v0'),
                                      augment_wrapper_args={"reward_scale": 1.0},
                                      agent_args={"batch_size": 3})

Experiment.register(BootstrappedDQNIceHockey, "Bootstrapped DQN for the IceHockey")


class RandomizedBootstrappedDQNBreakOut(BootstrappedDQNAtari):
    def __init__(self):
        import math
        from hobotrl.algorithms.bootstrapped_DQN import RandomizedBootstrappedDQN

        def eps_function(step):
            return 0.2*(math.cos(step/4.0e5*math.pi) + 1)

        BootstrappedDQNAtari.__init__(self,
                                      env=gym.make('Breakout-v0'),
                                      runner_args={"no_reward_reset_interval": 2000},
                                      agent_args={"eps_function": eps_function},
                                      agent_type=RandomizedBootstrappedDQN)

Experiment.register(RandomizedBootstrappedDQNBreakOut, "Randomized Bootstrapped DQN for the Breakout")


def demo_experiment_generator(experiment_class, checkpoint_file_name, frame_time=0.05):
    """
    Generate a demo experiment using "EnvRunner2".

    :param experiment_class: class of the experiment.
    :param checkpoint_file_name: file name of the checkpoint that should be loaded.
    :param frame_time: will be passed to the environment runner.
    :return: an experiment.
    """
    class BootstrappedDQNDemo(Experiment):
        def run(self, args):
            from hobotrl.environments import EnvRunner2

            experiment = experiment_class()
            env_runner = EnvRunner2(env=experiment.env,
                                    agent=experiment.agent,
                                    log_dir=args.logdir,
                                    frame_time=frame_time)
            env_runner.run_demo(checkpoint_file_name)

    BootstrappedDQNDemo.__name__ = experiment_class.__name__ + "Demo"
    return BootstrappedDQNDemo

Experiment.register(demo_experiment_generator(RandomizedBootstrappedDQNBreakOut, "60000.ckpt", frame_time=0.1), "Demo for the Breakout")
Experiment.register(demo_experiment_generator(BootstrappedDQNPong, "4092000.ckpt", frame_time=0.01), "Demo for the Pong")
Experiment.register(demo_experiment_generator(BootstrappedDQNBattleZone, "2232000.ckpt", frame_time=0.02), "Demo for the Battle Zone")
Experiment.register(demo_experiment_generator(BootstrappedDQNEnduro, "17000000.ckpt", frame_time=0.02), "Demo for the Enduro")
Experiment.register(demo_experiment_generator(BootstrappedDQNIceHockey, "23400000.ckpt", frame_time=0.04), "Demo for the Ice Hockey")


class CEMBootstrappedDQNSnakeGame(Experiment):
    def run(self, args):
        """
        Run the experiment.
        """
        from environments.snake import SnakeGame
        from hobotrl.algorithms.bootstrapped_DQN import CEMBootstrappedDQN
        from hobotrl.environments import EnvRunner2

        import os
        import random

        # Parameters
        random.seed(1105)  # Seed
        n_head = 10

        noise_candidates = [0.05, 0.10, 0.15, 0.20]
        portion_candidates = [0.3, 0.5, 0.8, 1.]
        grid = [(n, p) for n in noise_candidates for p in portion_candidates]

        for noise, portion in grid:

            log_dir = os.path.join(args.logdir, "%d_%d" % (noise*100, portion*10))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file_name = "booststrapped_DQN_Snake.csv"

            # Initialize the environment and the agent
            env = SnakeGame(3, 3, 1, 1, max_episode_length=30)
            agent = CEMBootstrappedDQN(observation_space=env.observation_space,
                                       action_space=env.action_space,
                                       reward_decay=1.,
                                       td_learning_rate=0.5,
                                       target_sync_interval=2000,
                                       nn_constructor=self.nn_constructor,
                                       loss_function=self.loss_function,
                                       trainer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize,
                                       replay_buffer_class=hrl.playback.MapPlayback,
                                       replay_buffer_args={"capacity": 20000},
                                       min_buffer_size=100,
                                       batch_size=20,
                                       n_heads=n_head,
                                       cem_noise=noise,
                                       cem_portion=portion,
                                       cem_update_interval=50)

            # Start training
            env_runner = EnvRunner2(env=env,
                                    agent=agent,
                                    n_episodes=1500,
                                    moving_average_window_size=100,
                                    no_reward_reset_interval=-1,
                                    checkpoint_save_interval=1000,
                                    log_dir=log_dir,
                                    log_file_name=log_file_name,
                                    render_env=False,
                                    render_interval=1000,
                                    render_length=200,
                                    frame_time=0.1,
                                    render_options={"mode": "ansi"}
                                    )
            env_runner.run()

    @staticmethod
    def loss_function(output, target):
        """
        Calculate the loss.
        """
        return tf.reduce_sum(tf.sqrt(tf.squared_difference(output, target)+1)-1, axis=-1)

    @staticmethod
    def nn_constructor(observation_space, action_space, n_heads, **kwargs):
        """
        Construct the neural network.
        """
        def leakyRelu(x):
            return tf.maximum(0.01*x, x)

        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

        def weight(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        def bias(shape):
            return tf.Variable(tf.constant(0.1, shape=shape))

        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        eshape = observation_space.shape
        nn_outputs = []

        # Layer 1 parameters
        n_channel1 = 8
        w1 = weight([3, 3, eshape[-1], n_channel1])
        b1 = bias([n_channel1])

        # Layer 2 parameters
        n_channel2 = 16
        w2 = weight([n_channel1*eshape[0]*eshape[1], n_channel2])
        b2 = bias([n_channel2])

        # Layer 1
        layer1 = leakyRelu(conv2d(x, w1) + b1)
        layer1_flatten = tf.reshape(layer1, [-1, n_channel1*eshape[0]*eshape[1]])

        # Layer 2
        layer2 = leakyRelu(tf.matmul(layer1_flatten, w2) + b2)

        nn_head_para = []

        for head in range(n_heads):
            with tf.variable_scope("head%d" % head) as scope_head:
                # Layer 3 parameters
                w3 = weight([n_channel2, 4])
                b3 = bias([4])

                # Layer 3
                layer3 = tf.matmul(layer2, w3) + b3

            nn_outputs.append(layer3)
            nn_head_para.append(tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=scope_head.name))

        return {"input": x,
                "head": nn_outputs,
                "head_para": nn_head_para}

Experiment.register(CEMBootstrappedDQNSnakeGame, "CEM Bootstrapped DQN for the Snake game")


class CEMBootstrappedDQNAtari(BootstrappedDQNAtari):
    @staticmethod
    def nn_constructor(observation_space, action_space, n_heads, **kwargs):
        """
        Construct the neural network.
        """
        def leakyRelu(x):
            return tf.maximum(0.01*x, x)

        import tensorflow.contrib.layers as layers
        nn_outputs = []

        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        print "input size:", x
        out = hrl.utils.Network.conv2d(input_var=x, h=8, w=8, out_channel=32,
                                       strides=[4, 4], activation=leakyRelu, padding="VALID", var_scope="conv1")
        # 20 * 20 * 32
        print "out size:", out
        out = hrl.utils.Network.conv2d(input_var=out, h=4, w=4, out_channel=64,
                                       strides=[2, 2], activation=leakyRelu, padding="VALID", var_scope="conv2")
        # 9 * 9 * 64
        print "out size:", out
        out = hrl.utils.Network.conv2d(input_var=out, h=3, w=3, out_channel=64,
                                       strides=[1, 1], activation=leakyRelu, padding="VALID", var_scope="conv3")

        # 7 * 7 * 64
        out = tf.reshape(out, [-1, int(np.product(out.shape[1:]))])
        out = layers.fully_connected(out, 512, activation_fn=leakyRelu)
        print "out size:", out

        nn_head_para = []
        for head in range(n_heads):
            with tf.variable_scope("head%d" % head) as scope_head:
                head = layers.fully_connected(out, action_space.n, activation_fn=None)

            nn_outputs.append(head)
            nn_head_para.append(tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=scope_head.name))

        return {"input": x,
                "head": nn_outputs,
                "head_para": nn_head_para}


class CEMBootstrappedDQNBreakout(CEMBootstrappedDQNAtari):
    def __init__(self):
        from hobotrl.algorithms.bootstrapped_DQN import CEMBootstrappedDQN
        super(CEMBootstrappedDQNBreakout, self).__init__(env=gym.make("BreakoutNoFrameskip-v0"),
                                                         agent_type=CEMBootstrappedDQN,
                                                         agent_args={"cem_noise": 0.05,
                                                                     "cem_portion": 0.8,
                                                                     "cem_update_interval": 50,
                                                                     "reward_decay": 0.99},
                                                         runner_args={"no_reward_reset_interval": 800})

Experiment.register(CEMBootstrappedDQNBreakout, "CEM Bootstrapped DQN for the Breakout")


class CEMBootstrappedDQNIceHockey(CEMBootstrappedDQNAtari):
    def __init__(self):
        from hobotrl.algorithms.bootstrapped_DQN import CEMBootstrappedDQN
        super(CEMBootstrappedDQNIceHockey, self).__init__(env=gym.make('IceHockey-v0'),
                                                          agent_type=CEMBootstrappedDQN,
                                                          agent_args={"cem_noise": 0.1,
                                                                      "cem_portion": 0.3,
                                                                      "cem_update_interval": 50})

Experiment.register(CEMBootstrappedDQNIceHockey, "CEM Bootstrapped DQN for the Ice Hockey")

Experiment.register(demo_experiment_generator(CEMBootstrappedDQNBreakout, "20600000.ckpt", frame_time=0.01), "Demo for the Breakout")
Experiment.register(demo_experiment_generator(CEMBootstrappedDQNIceHockey, "7000000.ckpt", frame_time=0.02), "Demo for the Breakout")


class PPOPendulum(PPOExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=2000,
                 discount_factor=0.9, entropy=3e-3, clip_epsilon=0.2,
                 epoch_per_step=1,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(3e-4),
                                                                           grad_clip=10.0),
                 batch_size=16,
                 horizon=256):
        if env is None:
            env = gym.make("Pendulum-v0")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        if f_create_net is None:
            dim_action = env.action_space.shape[-1]

            def f_net(inputs):
                l2 = 1e-4
                state = inputs[0]
                v = hrl.network.Utils.layer_fcs(state, [200, 100], 1, l2=l2, var_scope="v")
                v = tf.squeeze(v, axis=1)
                mean = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                   # activation_out=None,
                                                   activation_out=lambda x: tf.tanh(x / 4.0),
                                                   l2=l2, var_scope="mean")
                stddev = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                     # activation_out=None,
                                                     activation_out=lambda x: 4.0 * tf.sigmoid(x / 4.0),
                                                     l2=l2, var_scope="stddev")
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = f_net

        super(PPOPendulum, self).__init__(env, f_create_net, episode_n, discount_factor, entropy, clip_epsilon,
                                          epoch_per_step, network_optimizer_ctor, batch_size, horizon)
Experiment.register(PPOPendulum, "PPO for Pendulum")


class PPOBipedal(PPOExperiment):
    def __init__(self, env=None, f_create_net=None, episode_n=10000,
                 discount_factor=0.9, entropy=5e-4, clip_epsilon=0.2,
                 epoch_per_step=1,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(3e-4),
                                                                           grad_clip=10.0),
                 batch_size=16,
                 horizon=256):
        if env is None:
            env = gym.make("BipedalWalker-v2")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        if f_create_net is None:
            dim_action = env.action_space.shape[-1]

            def f_net(inputs):
                l2 = 1e-8
                state = inputs[0]
                v = hrl.network.Utils.layer_fcs(state, [200, 100], 1, l2=l2, var_scope="v")
                v = tf.squeeze(v, axis=1)
                mean = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                   # activation_out=None,
                                                   activation_out=lambda x: tf.tanh(x / 4.0),
                                                   l2=l2, var_scope="mean")
                stddev = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                     # activation_out=None,
                                                     activation_out=lambda x: 4.0 * tf.sigmoid(x / 4.0),
                                                     l2=l2, var_scope="stddev")
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = f_net

        super(PPOBipedal, self).__init__(env, f_create_net, episode_n, discount_factor, entropy, clip_epsilon,
                                          epoch_per_step, network_optimizer_ctor, batch_size, horizon)
Experiment.register(PPOBipedal, "PPO for BipedalWalker")


class ACBipedal(ACExperiment):

    def __init__(self, env=None, f_create_net=None, episode_n=20000, discount_factor=0.9,
                 entropy=CappedLinear(1e6, 1e-4, 1e-5),
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3),
                                                                           grad_clip=10.0), batch_size=16):
        if env is None:
            env = gym.make("BipedalWalker-v2")
            env = hrl.envs.AugmentEnvWrapper(env, reward_decay=discount_factor, reward_scale=0.1)
        if f_create_net is None:
            dim_action = env.action_space.shape[-1]

            def f_net(inputs):
                l2 = 1e-8
                state = inputs[0]
                v = hrl.network.Utils.layer_fcs(state, [200, 100], 1, l2=l2, var_scope="v")
                v = tf.squeeze(v, axis=1)
                mean = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                   # activation_out=None,
                                                   activation_out=lambda x: tf.tanh(x / 4.0),
                                                   l2=l2, var_scope="mean")
                stddev = hrl.network.Utils.layer_fcs(state, [200, 100], dim_action,
                                                     # activation_out=None,
                                                     activation_out=lambda x: 2.0 * tf.sigmoid(x / 4.0),
                                                     l2=l2, var_scope="stddev")
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = f_net
        super(ACBipedal, self).__init__(env, f_create_net, episode_n, discount_factor, entropy,
                                         network_optimizer_ctor, batch_size)
Experiment.register(ACBipedal, "actor critic for BipedalWalker")


if __name__ == '__main__':
    Experiment.main()
