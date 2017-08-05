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
import hobotrl.algorithms.dqn_c as dqn_c

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
                network_optmizer=n_optimizer,
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


class ADQNExperiment(Experiment):
    def run(self, args):
        env = gym.make('Pendulum-v0')
        env = hrl.envs.C2DEnvWrapper(env, [5])
        env = hrl.envs.ScaledRewards(env, 0.1)
        state_shape = list(env.observation_space.shape)

        discount_factor = 0.9

        def f_q(inputs):
            q = network.Utils.layer_fcs(inputs[0], [200, 100], env.action_space.n, l2=1e-4)
            return {"q": q}

        def create_optimizer():
            return tf.train.AdamOptimizer(1e-3)

        def create_agent(n_optimizer, global_step):
            agent = dqn_c.DQN(
                f_create_q=f_q,
                state_shape=state_shape,
                # OneStepTD arguments
                num_actions=env.action_space.n,
                discount_factor=0.99,
                ddqn=False,
                # target network sync arguments
                target_sync_interval=100,
                target_sync_rate=1.0,
                # sampler arguments
                update_interval=4, replay_size=1000, batch_size=32,
                # epsilon greedy arguments
                greedy_epsilon=hrl.utils.CappedLinear(1e5, 0.5, 0.1),
                global_step=global_step,
                network_optmizer=n_optimizer
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

Experiment.register(ADQNExperiment, "test ADQN")

if __name__ == '__main__':
    Experiment.main()
