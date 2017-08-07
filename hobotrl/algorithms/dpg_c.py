# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl
import hobotrl.network as network
import hobotrl.sampling as sampling
import hobotrl.target_estimate as target_estimate
import hobotrl.tf_dependent.distribution as distribution
from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.policy import OUExplorationPolicy
import hobotrl.async as async


class DPGUpdater(network.NetworkUpdater):
    def __init__(self, actor, critic, target_estimator, discount_factor):
        """

        :param actor:
        :type actor network.NetworkFunction
        :param critic:
        :type critic network.NetworkFunction
        :param target_estimator:
        :type target_estimator: target_estimate.TargetEstimator
        """
        super(DPGUpdater, self).__init__()
        self._actor, self._critic, self._target_estimator = \
            actor, critic, target_estimator
        self._dim_action = actor.output().op.shape.as_list()[-1]
        op_q = critic.output().op
        with tf.name_scope("DPGUpdater"):
            with tf.name_scope("input"):
                self._input_target_q = tf.placeholder(dtype=tf.float32, shape=[None], name="input_target_q")
                # self._input_target_action = tf.placeholder(dtype=tf.float32,
                #                                            shape=[None, self._dim_action],
                #                                            name="input_target_action")
                self._input_action_gradient = tf.placeholder(dtype=tf.float32,
                                                             shape=[None, self._dim_action],
                                                             name="input_action_gradient")
            with tf.name_scope("critic"):
                self._critic_loss = tf.reduce_mean(network.Utils.clipped_square(
                    self._input_target_q - op_q
                ))
            with tf.name_scope("actor"):
                # critic.inputs[1] is input_action
                self._action_gradient = tf.gradients(critic.output().op, critic.inputs[1])[0]
                self._gradient_func = network.NetworkFunction(
                    outputs=network.NetworkSymbol(self._action_gradient, "gradient", critic.network),
                    inputs=critic.inputs
                )
                self._actor_loss = tf.reduce_sum(actor.output().op * self._input_action_gradient, axis=1)
                self._actor_loss = -tf.reduce_mean(self._actor_loss)
                print "self._actor_loss, self._input_action_gradient", self._actor_loss, self._input_action_gradient

            self._op_loss = self._actor_loss + self._critic_loss
        self._update_operation = network.MinimizeLoss(self._op_loss,
                                                      var_list=self._actor.variables +
                                                               self._critic.variables)

    def declare_update(self):
        return self._update_operation

    def update(self, sess, batch, *args, **kwargs):
        state, action, reward, next_state, episode_done = batch["state"], \
                                                          batch["action"], \
                                                          batch["reward"], \
                                                          batch["next_state"], \
                                                          batch["episode_done"]
        target_q = self._target_estimator.estimate(state, action, reward, next_state, episode_done)
        current_action = self._actor(state)
        action_gradient = self._gradient_func(state, current_action)
        feed_dict = self._critic.input_dict(state, action)
        feed_dict.update(self._actor.input_dict(state))
        feed_dict.update({
            self._input_target_q: target_q,
            self._input_action_gradient: action_gradient
        })

        return network.UpdateRun(feed_dict=feed_dict, fetch_dict={
            "target_value": target_q,
            "actor_loss": self._actor_loss,
            "critic_loss": self._critic_loss,
            "loss": self._op_loss,
            "output_action": self._actor.output().op,
            "action_gradient": self._action_gradient
        })


class DPG(sampling.TransitionBatchUpdate,
          BaseDeepAgent):
    def __init__(self,
                 f_create_net, state_shape, dim_action,
                 # ACUpdate arguments
                 discount_factor, target_estimator=None,
                 # optimizer arguments
                 network_optimizer=None, max_gradient=10.0,
                 # policy arguments
                 ou_params=(0.0, 0.2, 0.2),
                 # target network sync arguments
                 target_sync_interval=10,
                 target_sync_rate=0.01,
                 # sampler arguments
                 sampler=None,
                 batch_size=32,
                 *args, **kwargs):
        kwargs.update({
            "f_create_net": f_create_net,
            "state_shape": state_shape,
            "dim_action": dim_action,
            "discount_factor": discount_factor,
            "target_estimator": target_estimator,
            "max_gradient": max_gradient,
            "batch_size": batch_size,
            "ou_params": ou_params,
        })
        if network_optimizer is None:
            network_optimizer = network.LocalOptimizer(grad_clip=max_gradient)
        if sampler is None:
            sampler = sampling.TransitionSampler(hrl.playback.MapPlayback(1000), batch_size, 4)
        kwargs.update({"sampler": sampler})
        super(DPG, self).__init__(*args, **kwargs)

        self._q_function = network.NetworkFunction(self.network["q"])
        self._actor_function = network.NetworkFunction(self.network["action"])
        self._target_q_function = network.NetworkFunction(self.network.target["q"])
        self._target_actor_function = network.NetworkFunction(self.network.target["action"])
        if target_estimator is None:
            target_estimator = target_estimate.ContinuousActionEstimator(
                self._target_actor_function, self._target_q_function, discount_factor)
        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(
            DPGUpdater(actor=self._actor_function,
                       critic=self._q_function,
                       target_estimator=target_estimator,
                       discount_factor=discount_factor), name="ac")
        network_optimizer.add_updater(network.L2(self.network), name="l2")
        network_optimizer.compile()

        self._policy = OUExplorationPolicy(self._actor_function, *ou_params)
        self._target_sync_interval = target_sync_interval
        self._target_sync_rate = target_sync_rate
        self._update_count = 0

    def init_network(self, f_create_net, state_shape, dim_action, *args, **kwargs):
        input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
        input_action = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name="input_action")
        return network.NetworkWithTarget([input_state, input_action],
                                         network_creator=f_create_net,
                                         var_scope="learn",
                                         target_var_scope="target")

    def update_on_transition(self, batch):
        self._update_count += 1
        self.network_optimizer.updater("ac").update(self.sess, batch)
        self.network_optimizer.updater("l2").update(self.sess)
        info = self.network_optimizer.optimize_step(self.sess)
        if self._update_count % self._target_sync_interval == 0:
            self.network.sync_target(self.sess, self._target_sync_rate)
        return info, {}

    def set_session(self, sess):
        super(DPG, self).set_session(sess)
        self.network.set_session(sess)

    def act(self, state, **kwargs):
        return self._policy.act(state, **kwargs)


if __name__ == '__main__':
    import gym
    import hobotrl.environments as envs

    env = gym.make('Pendulum-v0')
    env = hrl.envs.ScaledRewards(env, 0.1)
    state_shape = list(env.observation_space.shape)
    dim_action = env.action_space.shape[0]
    global_step = tf.get_variable(
        'global_step', [], dtype=tf.int32,
        initializer=tf.constant_initializer(0), trainable=False
    )

    def f_net(inputs):
        l2 = 1e-8
        state, action = inputs[0], inputs[1]
        actor = network.Utils.layer_fcs(state, [200, 100], dim_action, activation_out=tf.nn.tanh, l2=l2, var_scope="action")
        se = network.Utils.layer_fcs(state, [200], 100, activation_out=None, l2=l2, var_scope="se")
        se = tf.concat([se, action], axis=-1)
        q = network.Utils.layer_fcs(se, [100], 1, activation_out=None, l2=l2, var_scope="q")
        q = tf.squeeze(q, axis=1)
        return {"q": q, "action": actor}

    sampler = async.AsyncTransitionSampler(hrl.playback.MapPlayback(1000), 32)
    agent = DPG(
        f_create_net=f_net, state_shape=state_shape, dim_action=dim_action,
        # ACUpdate arguments
        discount_factor=0.9, target_estimator=None,
        # optimizer arguments
        network_optimizer=network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
        max_gradient=10.0,
        # policy arguments
        ou_params=(0, 0.2, 0.2),
        # target network sync arguments
        target_sync_interval=10,
        target_sync_rate=0.01,
        # sampler arguments
        sampler=sampler,
        batch_size=32,
        global_step=global_step,
    )
    agent = async.AsynchronousAgent(agent)
    #
    # def create_agent():
    #     return DPG(
    #         f_create_net=f_net, state_shape=state_shape, dim_action=dim_action,
    #         # ACUpdate arguments
    #         discount_factor=0.9, target_estimator=None,
    #         # optimizer arguments
    #         network_optimizer=network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
    #         max_gradient=10.0,
    #         # policy arguments
    #         ou_params=(0, 0.2, 0.2),
    #         # target network sync arguments
    #         target_sync_interval=10,
    #         target_sync_rate=0.01,
    #         # sampler arguments
    #         sampler=sampler,
    #         batch_size=32,
    #         global_step=global_step,
    #     )
    #
    # agent = async.AsynchronousAgent2(create_agent)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = agent.init_supervisor(
        graph=tf.get_default_graph(), worker_index=0,
        init_op=tf.global_variables_initializer(), save_dir="dpg_new"
    )
    with sv.managed_session(config=config) as sess:
        agent.set_session(sess)
        runner = hrl.envs.EnvRunner(
            env, agent, evaluate_interval=sys.maxint,
            render_interval=sys.maxint, logdir="dpg_new"
        )
        runner.episode(5000)
