#
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import logging

import tensorflow as tf
import gym

import hobotrl as hrl
from hobotrl.experiment import Experiment
from hobotrl.cluster import ClusterAgent
import hobotrl.network as network
import hobotrl.algorithms.ac_c as ac_c


class ACExperiment(Experiment):
    def run(self, args):
        env = gym.make('Pendulum-v0')
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.ScaledRewards(env, 0.1)
        state_shape = list(env.observation_space.shape)

        discount_factor = 0.9
        def f_net(inputs):
            l2 = 1e-4
            state = inputs[0]
            q = network.Utils.layer_fcs(state, [200, 100], env.action_space.n, l2=l2, var_scope="q")
            pi = network.Utils.layer_fcs(state, [200, 100], env.action_space.n, activation_out=tf.nn.softmax, l2=l2,
                                         var_scope="pi")
            return {"q": q, "pi": pi}

        def create_optimizer():
            return tf.train.AdamOptimizer(1e-3)

        def create_agent(n_optimizer, global_step):
            agent = ac_c.ActorCritic(
                f_create_net=f_net,
                state_shape=state_shape,
                # ACUpdate arguments
                num_actions=env.action_space.n,
                discount_factor=discount_factor,
                entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-2),
                target_estimator=None,
                max_advantage=100.0,
                # optimizer arguments
                network_optmizer=network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
                max_gradient=10.0,
                # sampler arguments
                sampler=None,
                batch_size=8,

                global_step=global_step,
            )
            return agent

        agent = ClusterAgent(create_agent, create_optimizer, args.cluster, args.job, args.index, args.logdir)
        with agent.wait_for_session() as sess:
            agent.set_session(sess)
            runner = hrl.envs.EnvRunner(env, agent, reward_decay=discount_factor,
                                        evaluate_interval=sys.maxint, render_interval=sys.maxint,
                                        render_once=True,
                                        logdir=args.logdir if args.index == 0 else None)
            runner.episode(1000)

Experiment.register(ACExperiment, "test A3C")

if __name__ == '__main__':
    Experiment.main()
