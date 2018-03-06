# -*- coding: utf-8 -*-
import sys
sys.append('../../')
import tensorflow as tf
import gym
from hobotrl.utils import CappedExp
from hobotrl.network import LocalOptimizer, Utils
from hobotrl.environments.environments import ScaledFloatFrame, FrameStack, ScaledRewards, NoneSkipWrapper
from test.exp_algorithms import Experiment, DPGExperiment
from test.exp_car import CarGrassWrapper, CarContinuousWrapper
from .env import CarRacingGoalWrapper


class DDPGCarRacingSoftGoal(DPGExperiment):
    def __int__(self, env=None, episode_n=1000, discount_factor=0.95,
                f_se=None, f_actor=None, f_critic=None, network_optimizer_ctor=None,
                target_sync_interval=10, target_sync_rate=0.01, batch_size=32,
                ou_params=None, replay_capacity=10000, **kwargs):
        # environment
        if env is None:
            env = gym.make("CarRacing-v0")
            env = CarGrassWrapper(env, grass_penalty=0.5)
            env = CarContinuousWrapper(env)
            env = ScaledFloatFrame(env)
            env = NoneSkipWrapper(env, skip=5)
            env = FrameStack(env, 4)
            env = ScaledRewards(env, 0.2)
            env = CarRacingGoalWrapper(env)

        # network
        l2 = 1e-7
        nonlinear = tf.nn.elu
        dim_se = 256
        dim_action = env.action_space.shape[-1]
        if f_se is None:
            def f(inputs):
                input_observation = inputs[0]
                se_conv = Utils.conv2ds(
                    input_observation, shape=[(8, 8, 4), (16, 4, 2), (32, 3, 2)],
                    out_flatten=True, activation=nonlinear,
                    l2=l2, var_scope="se_conv"
                )

                se_linear = Utils.layer_fcs(
                    se_conv, [256], dim_se,
                    activation_hidden=nonlinear, activation_out=None,
                    l2=l2, var_scope="se_linear"
                )
                return {"se": se_linear}
            f_se = f

        if f_actor is None:
            def f(inputs):
                se = inputs[0]
                action = Utils.layer_fcs(
                    se, [256, 256], dim_action,
                    activation_hidden=nonlinear, activation_out=tf.nn.tanh,
                    l2=l2, var_scope="actor"
                )
                return {"action": action}

            f_actor = f
        if f_critic is None:
            def f(inputs):
                se, action = inputs[0], inputs[1]
                se = tf.concat([se, action], axis=-1)
                q = Utils.layer_fcs(
                    se, [256, 256], 1,
                    activation_hidden=nonlinear, activation_out=None,
                    l2=l2, var_scope="q"
                )
                q = tf.squeeze(q, axis=1)
                return {"q": q}

            f_critic = f
        if network_optimizer_ctor is None:
            network_optimizer_ctor = lambda: LocalOptimizer(tf.train.AdamOptimizer(3e-5), grad_clip=10.0)

        if ou_params is None:
            ou_params = (0, 0.2, CappedExp(2e5, 0.5, 0.02))

        super(DDPGCarRacingSoftGoal, self).__init__(
            env, f_se, f_actor, f_critic, episode_n, discount_factor, network_optimizer_ctor,
            ou_params, target_sync_interval, target_sync_rate, batch_size, replay_capacity,
            **kwargs
        )
Experiment.register(DDPGCarRacingSoftGoal, "DDPG with soft goal.")

