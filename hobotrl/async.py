# -*- coding: utf-8 -*-
import sys
import time
import traceback
import logging
import Queue
import threading
import wrapt

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
        # global_device = "/job:ps/task:0"
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
                    self._global_network = global_agent.network

            for i in range(worker_n):
                with tf.device("/job:worker/task:%d" % i):
                    with tf.variable_scope("worker%d" % i):
                        n_optimizer = network.OptimizerPlaceHolder()
                        worker = agent_creator(n_optimizer, global_step)
                        local_network = worker.network
                        local_to_global = dict(zip(local_network.variables, self._global_network.variables))
                        n_optimizer.set_optimizer(
                            network.DistributedOptimizer(global_optimizer=global_optimizer,
                                                         local_global_var_map=local_to_global,
                                                         name="distributed_variable")
                        )
                    if i == job_index:
                        agent = worker

        self._agent = agent
        with tf.name_scope("init_local_weight"):
            self._init_local_weight = tf.group(*[tf.assign(var_l, var_g) for var_l, var_g in
                                                 zip(self._agent.network.variables, self._global_network.variables)])

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        return self._agent.step(state, action, reward, next_state, episode_done, **kwargs)

    def act(self, state, **kwargs):
        return self._agent.act(state, **kwargs)

    def new_episode(self, state):
        return self._agent.new_episode(state)

    def create_session(self, config=None, **kwargs):
        sess = self._agent.create_session(config=config,
                                          master=self._server.target,
                                          worker_index=self._job_index,
                                          save_dir=self._logdir,
                                          restore_var_list=self._global_network.variables)
        sess.run(self._init_local_weight)
        return sess

    def set_session(self, sess):
        self._agent.set_session(sess)

    @property
    def sess(self):
        return self._agent.sess


class AsynchronousAgent(wrapt.ObjectProxy):
    """Agent with async. training and inference thread.
    Creates a rate-controlled training thread for updating network. Training
    thread is monitored and re-spawned if stopped unexpectedly.
    """
    def __init__(self, agent, *args, **kwargs):
        super(AsynchronousAgent, self).__init__(agent)
        self._queue_step = Queue.Queue(maxsize=-1)
        self._queue_info = Queue.Queue(maxsize=-1)
        self._stop_monitor = threading.Event()
        self._stop_monitor.clear()

        self._trn_thread_monitor = threading.Thread(
            target=self.monitor_loop,
            args=(RateControlTrainingThread,
                  (self.__wrapped__, self._queue_step, self._queue_info),
                  kwargs, self._stop_monitor,)
        )
        self._trn_thread_monitor.start()

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        self._queue_step.put(
            {"args": (state, action, reward, next_state, episode_done),
             "kwargs": kwargs,}
        )
        info = {}
        # TODO: if only return the latest info, why not assume _infoq.size = 1?
        while self._queue_info.qsize() > 0:
            info = self._queue_info.get()
        return info

    def stop(self, *args, **kwargs):
        logging.warning(
            "[AsynchronousAgent.stop()]: stopping monitoring thread."
        )
        self._stop_monitor.set()
        self._trn_thread_monitor.join()

    def monitor_loop(self, class_trn_thread, args_trn_thread, kwargs_trn_thread,
                     stop_event, *args, **kwargs):

        _thread_train = class_trn_thread(*args_trn_thread, **kwargs_trn_thread)
        _thread_train.start()
        while not stop_event.is_set():
            if not _thread_train.is_stopped() and not _thread_train.is_alive():
                logging.warning(
                    "[AsynchronousAgent.monitor_loop()]: "
                    "training thread stopped unexpectedly, respawning..."
                )
                _thread_train = class_trn_thread(
                    *args_trn_thread, **kwargs_trn_thread)
                _thread_train.start()
            else:
                logging.warning(
                    "[AsynchronousAgent.monitor_loop()]: "
                    "training thread running okay!"
                )
            time.sleep(60.0)  # check trn thread status periodically
        logging.warning(
            "[AsynchronousAgent.monitor_loop()]: stopping training thread."
        )
        _thread_train.stop()
        _thread_train.join()
        logging.warning(
            "[AsynchronousAgent.monitor()]: quiting monitoring thread."
        )
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


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

    def create_session(self, config=None, save_dir=None, **kwargs):
        sess = self._step_agent.create_session(config=config,
                                               save_dir=save_dir)
        self._act_agent.set_session(sess)
        return sess

    @property
    def sess(self):
        return self._step_agent.sess


class TrainingThread(threading.Thread):

    def __init__(self, agent, step_queue, info_queue,
                 group=None, target=None, name=None, args=(),
                 kwargs=None, verbose=None):
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
        super(TrainingThread, self).__init__(
            group=group, target=target, name=name,
            args=args, kwargs=kwargs, verbose=verbose
        )
        self._agent = agent
        self._step_queue, self._info_queue = step_queue, info_queue
        self._stopped = False  # poisson pill
        self._first_sample_arrived = False
        self.__n_step = 0

    def run(self):
        while not self._stopped:
            try:
                self.step()
            except:
                print "[TrainingThread.run()]: step exception:"
                traceback.print_exc()
        logging.warning(
            "[TrainingThread.run()]: returning."
        )

    def step(self, *args, **kwargs):
        # get data from step queue
        queue_empty = self._step_queue.qsize() == 0
        if queue_empty:
            step = {
                "kwargs": {
                    "async_buffer_empty": True,
                },
                "args": (None, None, None, None, None)
            }
            time.sleep(0.001)
        else:
            step = self._step_queue.get(block=True)
            self._first_sample_arrived = True

        # async_buffer_end signal for asynchronous samplers, representing end of step queue
        # TODO: ??? aync_buffer end, async_buffer_empty?
        if self._first_sample_arrived:
            t = time.time()
            info = self._agent.step(
                *step["args"], async_buffer_end=queue_empty, **step["kwargs"]
            )
            t_learn = time.time() - t
            info['TrainingThread/t_step'] = t_learn
            self.__n_step += 1
            # update the item in info_queue with the latest info
            old_info = {}
            if self._info_queue.qsize() > 0:
                try:
                    # non-blocking invocation
                    old_info = self._info_queue.get(block=False)
                except:
                    pass
            info['TrainingThread/n_step'] = self.__n_step
            old_info.update(info)
            self._info_queue.put(old_info)
        return {}

    def stop(self):
        logging.warning(
            "[TrainingThread.step()]: setting poison pill."
        )
        self._stopped = True

    def is_stopped(self):
        return self._stopped

    @property
    def len_step_queue(self):
        return self._step_queue.qsize()


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
        if "async_buffer_empty" in kwargs and not kwargs["async_buffer_empty"]:
            self._replay.push_sample(self._sample_maker(state, action, reward, next_state, episode_done, **kwargs))
        self._step_n += 1
        async_buffer_end = kwargs["async_buffer_end"] or \
                           "async_buffer_empty" in kwargs and kwargs["async_buffer_empty"]
        if async_buffer_end and self._replay.get_count() >= self._minimum_count:
            return self._replay.sample_batch(self._batch_size)
        else:
            return None


class RateControlMixin(object):
    """Adaptive rate control for threads with step() method.
    Support three throttling methods:
        1. 'rate': assign quota to make step() runs with this rate on average.
        2. 'ratio': assign certain quota per item coming from step queue.
        3. 'best_effort': run as fast as possible.
    Assumes super class has a `step()` method. Also assume super class has a
    `len_step_queue` attribute to make `ratio` control work.
    """

    def __init__(self, *args, **kwargs):
        """Initialization.

        :param method: throttle method, 'rate', 'ratio', or 'best_effort'.
        :param rate:
        :param ratio:
        :param args:
        :param kwargs:
        """
        if 'method' in kwargs:
            method = kwargs['method']
            del kwargs['method']
        else:
            method = None
        if 'rate' in kwargs:
            rate = kwargs['rate']
            del kwargs['rate']
        else:
            rate = None
        if 'ratio' in kwargs:
            ratio = kwargs['ratio']
            del kwargs['ratio']
        else:
            ratio = None
        super(RateControlMixin, self).__init__(*args, **kwargs)

        # determine throttle method
        self.__method = method
        if self.__method == 'rate':
            if rate is None:
                self.__rate = 1.0
                print "[RateControlMixin.__init__()]: using default rate = 1.0."
            else:
                assert rate > 0
                self.__rate = rate
                print "[RateControlMixin.__init__()]: using rate = {}.".format(
                    rate)
            self.__t_last_call = time.time()
        elif self.__method == 'ratio':
            if ratio is None:
                self.__ratio = 1.0
                print "[RateControlMixin.__init__()]: using default ratio = 1.0."
            else:
                assert ratio > 0
                self.__ratio = ratio
                print "[RateControlMixin.__init__()]: using ratio = {}.".format(
                    ratio)
            self.__len_step_queue = 0
        else:
            self.__method = 'best_effort'
            print "[RateControlMixin.__init__()]: will run thread with best " \
                  "effort."

        # initialize quota
        self.__quota = 0
        self.__MAX_QUOTA = sys.maxsize

    def step(self, *args, **kwargs):
        len_step_queue = self.len_step_queue

        # adjust quota
        if self.__method == "ratio":
            self.__quota += self.__ratio * max(
                len_step_queue - self.__len_step_queue, 0
            )
        elif self.__method == 'rate':
            t = time.time()
            self.__quota += self.__rate * max(t - self.__t_last_call, 0)
            self.__t_last_call = t
        self.__quota = max(min(self.__quota, self.__MAX_QUOTA), 0)

        # throttle calls to the step() method of super class
        if self.__quota >= 1 or self.__method == 'best_effort':
            # logging.warning(
            #     "[RateControlMixin.step()]: "
            #     "got quota {} to step once @ {}.".format(
            #         self.__quota, len_step_queue
            #     )
            # )
            info = super(RateControlMixin, self).step(*args, **kwargs)
            self.__quota -= 1
            # if self.__quota < 1:
            #     logging.warning(
            #         "[RateControlMixin.step()]: "
            #         "emptied quota, quota ={}".format(self.__quota)
            #     )
        else:
            info = {}
            time.sleep(0.05)

        if self.__method == "ratio":
            self.__len_step_queue = len_step_queue

        return info


class RateControlTrainingThread(RateControlMixin, TrainingThread):
    def __init__(self, *args, **kwargs):
        super(RateControlTrainingThread, self).__init__(*args, **kwargs)

