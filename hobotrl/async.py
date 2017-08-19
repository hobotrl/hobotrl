# -*- coding: utf-8 -*-


import logging
import threading
import Queue

import tensorflow as tf
from core import Agent
import network
from sampling import TransitionSampler


class ClusterAgent(Agent):
    """
    Creates agents according to cluster_spec information,
    and wraps local agents according to job/job_index specification
    """
    def __init__(self, agent_creator, optimizer_creator,
                 cluster_spec, job, job_index, logdir,
                 grad_clip=None, *args, **kwargs):
        """

        :param agent_creator: function: agent_creator(tf.Variable global_step, NetworkOptimizer network_optimizer) => Agent
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


class AsynchronousAgent(Agent):

    def __init__(self, agent, *args, **kwargs):
        super(AsynchronousAgent, self).__init__(*args, **kwargs)
        self._agent = agent
        self._queue = Queue.Queue(maxsize=-1)
        self._infoq = Queue.Queue(maxsize=-1)
        self._thread = TrainingThread(self._agent, self._queue, self._infoq)
        self._thread.start()

    def new_episode(self, state):
        return self._agent.new_episode(state)

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        self._queue.put({"kwargs": kwargs, "args": (state, action, reward, next_state, episode_done)})
        info = {}
        while self._infoq.qsize() > 0:
            info = self._infoq.get()
        return info

    def act(self, state, **kwargs):
        return self._agent.act(state, **kwargs)

    def set_session(self, sess):
        self._agent.set_session(sess)

    def init_supervisor(self, graph=None, worker_index=0, init_op=None, save_dir=None):
        return self._agent.init_supervisor(graph=graph, worker_index=worker_index, init_op=init_op, save_dir=save_dir)

    def stop(self, blocking=True):
        self._thread.stop()
        if blocking:
            self._thread.join()

    @property
    def sess(self):
        return self._agent.sess


class AsynchronousAgent2(Agent):

    def __init__(self, agent_creator, act_device="cpu:0", step_device="cpu:1", *args, **kwargs):
        super(AsynchronousAgent2, self).__init__(*args, **kwargs)
        with tf.device(act_device):
            with tf.variable_scope("act"):
                self._act_agent = agent_creator()
        with tf.device(step_device):
            with tf.variable_scope("step"):
                self._step_agent = agent_creator()
        self._syncer = network.NetworkSyncer(self._step_agent.network, self._act_agent.network)

        self._queue = Queue.Queue(maxsize=-1)
        self._infoq = Queue.Queue(maxsize=-1)
        self._thread = TrainingThread(self._step_agent, self._queue, self._infoq)
        self._thread.start()
        self._act_count = 0

    def new_episode(self, state):
        self._act_agent.new_episode(state)
        return self._step_agent.new_episode(state)

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        self._queue.put({"kwargs": kwargs, "args": (state, action, reward, next_state, episode_done)})
        info = {}
        while self._infoq.qsize() > 0:
            info = self._infoq.get()
        return info

    def act(self, state, **kwargs):
        self._act_count += 1
        if self._act_count % 1000 == 0:
            self._syncer.sync(self.sess, 1.0)

        return self._act_agent.act(state, **kwargs)

    def set_session(self, sess):
        self._step_agent.set_session(sess)
        self._act_agent.set_session(sess)

    def init_supervisor(self, graph=None, worker_index=0, init_op=None, save_dir=None):
        return self._step_agent.init_supervisor(graph=graph, worker_index=worker_index, init_op=init_op, save_dir=save_dir)

    @property
    def sess(self):
        return self._step_agent.sess


class TrainingThread(threading.Thread):

    def __init__(self, agent, step_queue, info_queue, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        """

        :param agent:
        :type agent: Agent
        :param step_queue: queue filled by AsynchronousAgent with step()s for inner agent's step()
        :type step_queue: Queue.Queue
        :param info_queue: queue filled by this class from inner agent's step() returns
                for AsynchronousAgent to pass back as the return value of AsynchronousAgent.step()
        :type info_queue: Queue.Queue
        :param group:
        :param target:
        :param name:
        :param args:
        :param kwargs:
        :param verbose:
        """
        super(TrainingThread, self).__init__(group, target, name, args, kwargs, verbose)
        self._agent, self._step_queue, self._info_queue = agent, step_queue, info_queue
        self._stopped = False

    def run(self):
        while not self._stopped:
            step = self._step_queue.get(block=True)
            queue_empty = self._step_queue.qsize() == 0
            # async_buffer_end signal for asynchronous samplers, representing end of step queue
            info = self._agent.step(*step["args"], async_buffer_end=queue_empty, **step["kwargs"])
            self._info_queue.put(info)

    def stop(self):
        self._stopped = True


class AsyncTransitionSampler(TransitionSampler):
    def __init__(self, replay_memory, batch_size, minimum_count=None, sample_maker=None):
        super(AsyncTransitionSampler, self).__init__(replay_memory, batch_size, 4, minimum_count, sample_maker)

    def step(self, state, action, reward, next_state, episode_done, **kwargs):
        """
        optional kwargs: async_buffer_end: True of event queue is empty
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :param kwargs:
        :return:
        """
        self._step_n += 1
        self._replay.push_sample(self._sample_maker(state, action, reward, next_state, episode_done, **kwargs))
        async_buffer_end = kwargs["async_buffer_end"]
        if async_buffer_end and self._replay.get_count() >= self._minimum_count:
            return self._replay.sample_batch(self._batch_size)
        else:
            return None
