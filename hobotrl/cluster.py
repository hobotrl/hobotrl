# -*- coding: utf-8 -*-

import logging

import tensorflow as tf
from core import Agent
import network


class ClusterAgent(Agent):
    """
    Creates agents according to cluster_spec information, and wraps local agents according to job/job_index specification
    """
    def __init__(self, agent_creator, optimizer_creator, cluster_spec, job, job_index, logdir, *args, **kwargs):
        """

        :param agent_creator: function: agent_creator(tf.Variable global_step, NetworkOptimizer n_optimizer) => Agent
        :param optimizer_creator: function: optimizer_creator() => tf.train.Optimizer
        :param cluster_spec:
        :param job:
        :param job_index:
        :param args:
        :param kwargs:
        """
        super(ClusterAgent, self).__init__(*args, **kwargs)
        self._job, self._job_index, self._logdir = job, job_index, logdir
        cluster = eval(cluster_spec)
        cluster_spec = tf.train.ClusterSpec(cluster)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._server = tf.train.Server(cluster_spec,
                                       job_name=job,
                                       task_index=job_index,
                                       config=config)
        worker_n = len(cluster["worker"])
        global_device = "/job:worker/task:0"
        if job == "ps":
            logging.warning("starting ps server")
            # thread will stuck here
            self._server.join()
        else:
            with tf.device(global_device):
                global_step = tf.get_variable('global_step', [],
                                              dtype=tf.int32,
                                              initializer=tf.constant_initializer(0),
                                              trainable=False)
                with tf.variable_scope("global"):
                    n_optimizer = network.OptimizerPlaceHolder()
                    global_optimizer = optimizer_creator()
                    global_agent = agent_creator(n_optimizer, global_step)
                    global_network = global_agent.network

            for i in range(worker_n):
                with tf.device("/job:worker/task:%d" % i):
                    with tf.variable_scope("worker%d" % i):
                        n_optimizer = network.OptimizerPlaceHolder()
                        worker = agent_creator(n_optimizer, global_step)
                    if i == job_index:
                        agent = worker
                        local_network = agent.network
                        local_to_global = dict(zip(local_network.variables, global_network.variables))
                        n_optimizer.set_optimizer(
                            network.DistributedOptimizer(global_optimizer=global_optimizer,
                                                         local_global_var_map=local_to_global,
                                                         name="distributed_variable"))

        self._agent = agent

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        return self._agent.step(state, action, reward, next_state, episode_done, **kwargs)

    def act(self, state, **kwargs):
        return self._agent.act(state, **kwargs)

    def new_episode(self, state):
        return self._agent.new_episode(state)

    def wait_for_session(self):
        sv = self._agent.init_supervisor(graph=tf.get_default_graph(), worker_index=self._job_index,
                                         init_op=tf.global_variables_initializer(), save_dir=self._logdir)
        return sv.prepare_or_wait_for_session(self._server.target)

    def set_session(self, sess):
        self._agent.set_session(sess)

    @property
    def sess(self):
        return self._agent.sess