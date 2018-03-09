#
# -*- coding: utf-8 -*-


import sys

from hobotrl.experiment import ParallelGridSearch

sys.path.append(".")

from OpenGL import GL
import gym
import roboschool
from exp_algorithms import *
import hobotrl.environments as envs


class StateStack(gym.Wrapper):
    def __init__(self, env, k=2):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.states = deque([], maxlen=k)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=old_space.low[0],
            high=old_space.high[0],
            shape=(old_space.shape[0] * k, ))

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.states.append(ob)
        return self._get_ob()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.states.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.states) == self.k
        return np.concatenate(list(self.states), axis=0)


class ReacherEndTorch(gym.Wrapper):

    def __init__(self, env, end_torch_penalty=1.0, speed_penalty=1.0):
        super(ReacherEndTorch, self).__init__(env)
        self._end_torch_penalty = end_torch_penalty
        self._speed_penalty = speed_penalty

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        torch_penalty = self.torch_penalty(observation, action)
        speed_penalty = self.speed_penalty(observation, action)
        return observation, reward + torch_penalty + speed_penalty, done, info

    def torch_penalty(self, observation, action):
        p = 0
        if observation[7] < -1.0 and action[1] < 0:
            p = self._end_torch_penalty * action[1]
        elif observation[7] > 1.0 and action[1] > 0:
            p = -self._end_torch_penalty * action[1]
        torch_threshold = 0.5
        torch = np.abs(action)
        p_torch = -np.sum((torch > torch_threshold) * (torch - torch_threshold))
        if p != 0 or p_torch != 0:
            logging.warning("end torch penalty:%s, torch penalty:%s", p, p_torch)
        return p + p_torch

    def speed_penalty(self, observation, action):
        p = 0
        speed_threshold = 1.0
        if observation[6] > speed_threshold and action[0] > 0:
            p = -self._speed_penalty * (action[0] + observation[6] - speed_threshold)
        elif observation[6] < -speed_threshold and action[0] < 0:
            p = self._speed_penalty * (action[0] + observation[6] + speed_threshold)
        if p != 0:
            logging.warning("speed penalty:%s", p)
        return p


class ScalePenalty(gym.RewardWrapper):

    def __init__(self, env, scale=1.0):
        super(ScalePenalty, self).__init__(env)
        self._scale = scale

    def _reward(self, reward):
        reward = reward * self._scale if reward < 0 else reward
        return reward


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


class PPOAnt(PPOExperiment):

    def __init__(self, env=None, f_create_net=None, episode_n=10000, discount_factor=0.9,
                 entropy=hrl.utils.CappedLinear(1e5, 1e-4, 1e-4),
                 clip_epsilon=0.1,
                 epoch_per_step=4,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0),
                 batch_size=16,
                 horizon=200):

        if env is None:
            env = gym.make("RoboschoolAnt-v1")
            env = envs.ScaledRewards(env, 0.1)
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
                                                     activation_out=lambda x: 4.0 * tf.sigmoid(x / 8.0),
                                                     l2=l2, var_scope="stddev")
                return {"v": v, "mean": mean, "stddev": stddev}
            f_create_net = f_net

        super(PPOAnt, self).__init__(env, f_create_net, episode_n, discount_factor, entropy, clip_epsilon,
                                     epoch_per_step, network_optimizer_ctor, batch_size, horizon)
Experiment.register(PPOAnt, "PPO for ant")


class PPOAntSearch(ParallelGridSearch):
    def __init__(self):
        super(PPOAntSearch, self).__init__(PPOAnt, parameters={
            "episode_n": [1000],
            "entropy": [1e-3, 1e-4, 1e-5, 1e-6],
            "clip_epsilon": [0.1, 0.2]
        }, parallel=4)
Experiment.register(PPOAntSearch, "grid search for PPO for ant")


class DPGAnt(DPGExperiment):

    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None,
                 episode_n=10000,
                 discount_factor=0.9,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0),
                 ou_params=(0, 0.2, hrl.utils.CappedExp(2e5, 0.5, 0.01)),
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 batch_size=128,
                 replay_capacity=100000, **kwargs):
        if env is None:
            env = gym.make("RoboschoolAnt-v1")
            env = MaxAndSkipEnv(env, max_len=1, skip=2)
            env = envs.ScaledRewards(env, 0.1)
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
                actor = hrl.network.Utils.layer_fcs(se, [200, 100], dim_action, activation_out=tf.nn.tanh, l2=l2,
                                                    var_scope="action")
                return {"action": actor}
            f_actor = f
        if f_critic is None:
            def f(inputs):
                se, action = inputs[0], inputs[1]
                se = tf.concat([se, action], axis=-1)
                q = hrl.network.Utils.layer_fcs(se, [100], 1, activation_out=None, l2=l2, var_scope="q")
                q = tf.squeeze(q, axis=1)
                return {"q": q}
            f_critic = f
        super(DPGAnt, self).__init__(env, f_se, f_actor, f_critic, episode_n, discount_factor, network_optimizer_ctor,
                                     ou_params, target_sync_interval, target_sync_rate, batch_size, replay_capacity, **kwargs)
Experiment.register(DPGAnt, "DPG for Ant")


class PPOReacher(PPOAnt):

    def __init__(self, env=None, f_create_net=None, episode_n=1000, discount_factor=0.9,
                 entropy=hrl.utils.CappedLinear(1e5, 1e-6, 1e-8),
                 clip_epsilon=0.1,
                 epoch_per_step=4,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0),
                 batch_size=16,
                 horizon=1000):
        if env is None:
            env = gym.make("RoboschoolReacher-v1")
            env = envs.ScaledRewards(env, 0.2)
        super(PPOReacher, self).__init__(env, f_create_net, episode_n, discount_factor, entropy, clip_epsilon,
                                         epoch_per_step, network_optimizer_ctor, batch_size, horizon)
Experiment.register(PPOReacher, "PPO for reacher")


class DPGReacher(DPGAnt):

    def __init__(self, env=None, f_se=None, f_actor=None, f_critic=None, episode_n=2000,
                 discount_factor=0.9,
                 network_optimizer_ctor=lambda: hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-4),
                                                                           grad_clip=10.0),
                 # ou_params=(0, 0.2, [hrl.utils.CappedExp(1e5, 0.5, 0.02),
                 #                     hrl.utils.CappedExp(1e6, 2.0, 0.02)]),
                 ou_params=(0, 0.2, hrl.utils.CappedExp(1e5, 0.5, 0.02)),
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 batch_size=128,
                 replay_capacity=100000, **kwargs):
        if env is None:
            env = gym.make("RoboschoolReacher-v1")
            # env = StateStack(env, k=2)
            # env = MaxAndSkipEnv(env, max_len=1, skip=2)
            env = ReacherEndTorch(env)
            # env = ScalePenalty(env, scale=2.0)
            env = envs.ScaledRewards(env, 0.2)
        super(DPGReacher, self).__init__(env, f_se, f_actor, f_critic, episode_n, discount_factor,
                                         network_optimizer_ctor, ou_params, target_sync_interval, target_sync_rate,
                                         batch_size, replay_capacity, **kwargs)
Experiment.register(DPGReacher, "DPG for reacher")


class DPGReacherSearch(ParallelGridSearch):

    def __init__(self):
        super(DPGReacherSearch, self).__init__(DPGReacher, {
            "episode_n": [10],
            "ou_params": [(0, 0.2, hrl.utils.CappedLinear(2e5, 0.5, 0.05)),
                          (0, 0.2, hrl.utils.CappedLinear(4e5, 0.5, 0.05)),
                          (0, 0.2, hrl.utils.CappedLinear(4e5, 0.5, 0.1)),
                          (0, 0.2, hrl.utils.CappedLinear(4e5, 0.2, 0.05))
                          ],
            "_r": [0, 1, 2]
        }, parallel=4)
Experiment.register(DPGReacherSearch, "grid search for dpg for reacher")


if __name__ == '__main__':
    Experiment.main()
