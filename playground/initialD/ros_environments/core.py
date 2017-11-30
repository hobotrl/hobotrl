# -*- coding: utf-8 -*-
"""Gym-like environment wrapper for Hobot Driving Simulator.

:file_name: core.py
:author: Jingchu Liu
:data: 2017-09-05
"""

# Basic python
import importlib
import signal
import time
import traceback
# Threading and Multiprocessing
import threading
import subprocess
import multiprocessing
from multiprocessing import JoinableQueue as Queue
from multiprocessing import Value, Event
# Image
from cv_bridge import CvBridge
# ROS
import rospy
from rospy.timer import Timer
from std_msgs.msg import Bool
from message_composer import MetaMessageComposer
import utils.message_filters as message_filters

# the persisitent part
class DrivingSimulatorEnv(object):
    """Environment wrapper for Hobot Driving Simulator.

    This class wraps the Hobot Driving Simulator with a Gym-like interface.
    Since Gym-like environments are driven by calls to the `step()` function
    while the simulator is driven by its internal clocks, we use a FIFO queue
    to fasilitate clock domain crossing. Specifically, the simulator backend and
    the caller of `step()` talks through a FIFO queue: the simulator writes
    observation and reward messages to the queue and reads action from the
    queue; while the caller reads observation and reward messages from the
    queue and write actions to the queue.

    Play action in queue with a fixed frequency to the backend.

    :param defs_obs:
    :param defs_reward:
    :param defs_action:
    :param rate_action:
    :param window_sizes: number of samples to take from buffer for observation
        and reward in each `step()` and `reset()`.
    :param buffer_sizes: number of samples to keep in buffer for observation
        and reward.
    :param func_compile_obs: function used to process observation taken from
        buffer before returning to caller. Default do nothing.
    :param func_compile_reward: function used to process reward taken from
        buffer before returning to caller. Default do nothing.
    :param func_compile_action: function used to process action before putting
        into buffer. Default do nothing.
    :param step_delay_target: the desired delay for env steps. Used to ensure
        we are sampling the backend at a constant pace.
    :param is_dummy_action: if True use rule-based action and ignore the agent
        action passed in during `step()`.
    :param backend_cmds: list of commands for setting up simulator backend.
        Each command is a list of strings for a POpen() object.
    """
    def __init__(self,
                 defs_obs, defs_reward, defs_action, rate_action,
                 window_sizes, buffer_sizes,
                 func_compile_obs=None,
                 func_compile_reward=None,
                 func_compile_action=None,
                 step_delay_target=None,
                 is_dummy_action=False,
                 backend_cmds=None,
                 node_timeout=20):
        """Initialization."""
        # params
        self.defs_obs = self.__import_defs(defs_obs)
        if func_compile_obs is not None:
            self.__compile_obs = lambda *args: func_compile_obs(*args)
        else:
            self.__compile_obs = lambda x: x
        self.len_obs = window_sizes['obs']

        self.defs_reward = self.__import_defs(defs_reward)
        if func_compile_reward is not None:
            self.__compile_reward = lambda *args: func_compile_reward(*args)
        else:
            self.__compile_reward = lambda x: x
        self.len_reward = window_sizes['reward']

        self.action_msg_composers = [MetaMessageComposer(ac[1]) for ac in defs_action]
        self.defs_action = self.__import_defs(defs_action)
        if func_compile_action is not None:
            self.__compile_action = lambda *args: func_compile_action(*args)
        else:
            self.__compile_action = lambda x: x
        self.rate_action = rate_action
        self.is_dummy_action = is_dummy_action

        self.buffer_sizes = buffer_sizes
        self.MAX_EXCEPTION = 100  # maximum number of q exceptions per episode.
        self.NODE_TIMEOUT = node_timeout

        self.STEP_DELAY_TARGET = step_delay_target  # target delay for each step()
        self.last_step_t = None
        self.phase_target = 0.0
        self.phase_err = None
        self.phase_err_i = 0.0
        self.phase_delta = 0.0

        # inter-process flags (mp.Event)
        self.is_exiting = Event()
        self.is_backend_up = Event()    # if backend process is up and running
        self.is_q_ready = Event()       # if qs are ready
        self.is_q_cleared = Event()     # if qs from prevoius session is cleared
        self.is_envnode_up = Event()    # if env node is up
        self.is_envnode_terminatable = Event()  # if env node should be killed
        self.is_envnode_resetting = Event()  # if env node is undergoing reset
        self.is_env_resetting = Event()  # if environment is undergoing reset
        self.is_env_done = Event()  # if environment is is done for this ep
        self.is_env_done.set()
        self.is_receiving_obs = Event()  # if obs topic are publishing
        self.cnt_q_except = Value('i', self.MAX_EXCEPTION)

        # backend
        self.backend_cmds = backend_cmds if backend_cmds is not None else []
        self.proc_backend = []

        # monitor threads 
        self.thread_node_monitor = threading.Thread(target=self.__node_monitor)
        self.thread_node_monitor.start()

    def step(self, action):
        """Evolve environment forward by one step, wraps around a __step()
        method."""
        # check of env is properly reset.
        if self.is_env_done.is_set():
            raise Exception('[step()]: Env is done, please reset first!')

        if self.is_env_resetting.is_set():
            raise Exception('[step()]: reset() is in progress and not returned.')

        # step delay and phase regularization
        delay = time.time() - self.last_step_t
        delay_delta = self.STEP_DELAY_TARGET - delay + self.phase_delta

        if self.STEP_DELAY_TARGET is not None:
            if delay_delta > 0:
                time.sleep(delay_delta)
            else:
                pass
                print (
                    "[step()]: delay delta < 0. Delay {:.3f}/{:.3f}, Phase {:.5f}/{:.5f}."
                ).format(
                           delay, self.STEP_DELAY_TARGET,
                           self.phase_err, self.phase_target
                )

        # set point
        self.last_step_t = time.time()
        phase_err = \
            (self.phase_target - self.last_step_t + self.STEP_DELAY_TARGET/2) \
            % self.STEP_DELAY_TARGET - self.STEP_DELAY_TARGET/2
        self.phase_err_i += phase_err
        phase_err_d = (phase_err - self.phase_err) if self.phase_err is not None else 0
        self.phase_err = phase_err
        self.phase_delta = 0.1*(self.phase_err + self.phase_err_i + phase_err_d)

        # build action ROS msg
        # Note: users need to make sure the ROS msg can be initialized with the
        # data passed in.
        action = self.__compile_action(action)
        new_action = []
        for i in range(len(self.defs_action)):
            new_action.append(self.action_msg_composers[i](action[i]))
        action = tuple(new_action)

        # do __step(), every 10 fails try reset
        num_fail = 10
        while True:
            ret = self.__step(action)
            if ret is not None:
                break
            else:
                num_fail -= 1
                if num_fail == 0:
                    print "[step()]: __step failed for 10 times, try reset()."
                    self.reset()
                    num_fail = 10
            time.sleep(0.5)
        next_state, reward, done, info = ret
        #print "[step()]: action {}, reward {}, done {}.".format(
        #    action, reward, done)

        # set done
        if done:
            self.is_env_done.set()

        step_self_delay = time.time() - self.last_step_t
        #print "[step()]: step self delay {}".format(step_self_delay)

        return next_state, reward, done, info

    def __step(self, action):
        """Evolve environment by one step.

        Interacts with the backend simulator through multiple queues. Put
        the latest action into queue and extract observations, rewards, and
        done signal from queues.

        Catch exceptions when interacting with queues:
            1) modify queue counter for queue monitor to shut node down if
            there are too many exceptions. The exception counter is decreased
            by 1 per exception and increased per sucessful __step.
            2) return None for the unsuccessful interactions.

        Return None if failed to grep data from backend.
        """
        # wait until envnode, q, and backend is up and running
        # TODO: this while loop may trap the program in a deadlock, add a fail cnt.
        cnt_fail = 30
        while True:
            if self.is_backend_up.is_set() and \
               self.is_q_ready.is_set() and \
               self.is_envnode_up.is_set() and \
               self.is_receiving_obs.is_set():
                break
            else:
                print "[__step()]: backend {}, node {}, queue {}, obs {}.".format(
                   self.is_backend_up.is_set(),
                   self.is_envnode_up.is_set(),
                   self.is_q_ready.is_set(),
                   self.is_receiving_obs.is_set()
                )
                cnt_fail -= 1
                with self.cnt_q_except.get_lock():
                    if self.cnt_q_except.value<=0:
                        return None
                if cnt_fail == 0:
                    print "[__step()]: wait timeout, return None."
                    return None
                time.sleep(0.5)

        # action
        while True:
            try:
                if self.q_action.full():
                    self.q_action.get(timeout=0.05)
                    self.q_action.task_done()
                self.q_action.put(action, timeout=0.05)
                break
            except:
                with self.cnt_q_except.get_lock():
                    if self.cnt_q_except.value>0:
                        self.cnt_q_except.value -= 1
                    else:
                        return None
                    print "[__step()]: exception putting action into queue. {}.".format(
                        self.cnt_q_except.value)
                time.sleep(0.1)

        # observation
        obs_list = []
        for _ in range(self.len_obs):
            while True:
                try:
                    next_states = self.q_obs.get(timeout=0.05)
                    self.q_obs.task_done()
                    break
                except:
                    with self.cnt_q_except.get_lock():
                        if self.cnt_q_except.value>0:
                            self.cnt_q_except.value -= 1
                        else:
                            return None
                        #print "[__step()]: exception getting observation. {}.".format(
                        #    self.cnt_q_except.value)
                    time.sleep(0.1)
            obs_list.append(next_states)
        next_state = self.__compile_obs(obs_list)

        # reward
        reward_list = []
        for _ in range(self.len_reward):
            while True:
                try:
                    rewards = self.q_reward.get(timeout=0.05)
                    self.q_reward.task_done()
                    break
                except:
                    with self.cnt_q_except.get_lock():
                        if self.cnt_q_except.value>0:
                            self.cnt_q_except.value -= 1
                        else:
                            return None
                        #print "[__step()]: exception getting reward. {}.".format(
                        #    self.cnt_q_except.value)
                    time.sleep(0.1)
            reward_list.append(rewards)
        # p_str = "[step()]: reward vector "
        # fmt_dict = {float: '{:.4f},', bool: '{},', int: "{},"}
        # for reward in reward_list:
        #     slice_str = " ".join(
        #         map(lambda ele: fmt_dict[type(ele)], reward)).format(*reward)
        #     p_str += "\n    [" + slice_str + "],"
        # print p_str
        reward = self.__compile_reward(reward_list)

        # done
        # if True:
        while True:
            try:
                done = self.q_done.get(timeout=0.05)
                self.q_done.task_done()
                break
            except:
                with self.cnt_q_except.get_lock():
                    if self.cnt_q_except.value>0:
                        self.cnt_q_except.value -= 1
                    else:
                        return None
                    print "[__step()]: exception getting done. {}.".format(self.cnt_q_except.value)
                time.sleep(0.1)

        # info
        info = {}

        # increase counter by `credit` per successfuly step
        credit = 1 + self.len_obs + self.len_reward + 1
        with self.cnt_q_except.get_lock():
            while credit>0 and self.cnt_q_except.value<self.MAX_EXCEPTION:
                self.cnt_q_except.value += 1
                credit -= 1

        return next_state, reward, done, info

    def reset(self, **kwargs):
        while True:
            ret = self.__reset(**kwargs)
            if ret is not None:
                return ret
            else:
                print "[reset()]: reset faiiled retry in one sec."
                time.sleep(1.0)

    def __reset(self, **kwargs):
        """Environment reset."""
        # Setting sync. events
        # 1. Setting env_resetting will block further call to step.
        #    Although in normal process step is only called after a successful
        #    reset call, we still enforce this as an additional safety.
        self.is_env_resetting.set()
        # 2. Setting backend_up and clearing q_ready will immediately block
        #    subsequent queue access here in reset, until a new envnode
        #    resetting procedure is complete.
        #    Setting envnode_terminatable will force frontend process to return
        #    and Queue monitor to clear q_ready event and empty queues.
        #    Note the above event manipulation is only performed if the env
        #    node is not inside a resetting procesure. This is because these
        #    event will also be set or cleared in such procedures and there may
        #    be conflict in also doing so here.
        if not self.is_envnode_resetting.is_set():
            if self.is_backend_up.is_set():
                self.is_backend_up.clear()
            if self.is_q_ready.is_set():
                self.is_q_ready.clear()
            if self.is_envnode_up.is_set():
                self.is_envnode_terminatable.set()
        # 3. Clearing env_done event means a green light for node_monitor
        #    process to set up a new front-end node
        self.is_env_done.clear()


        # wait until backend is up
        cnt_fail = 30
        while True:
            if self.is_backend_up.is_set() and \
               self.is_q_ready.is_set() and \
               self.is_envnode_up.is_set() and \
               self.is_receiving_obs.is_set():
                   break
            else:
                print "[reset]: backend {}, node {}, queue {}, obs {}, waiting.".format(
                    self.is_backend_up.is_set(),
                    self.is_envnode_up.is_set(),
                    self.is_q_ready.is_set(),
                    self.is_receiving_obs.is_set()
                )
                with self.cnt_q_except.get_lock():
                    if self.cnt_q_except.value<=0:
                        return None
                cnt_fail -= 1
                if cnt_fail == 0:
                    print "[__reset()]: wait timeout, return None."
                    return None
                time.sleep(0.5)

        # start step delay clock and correct phase
        phase_err = (self.phase_target - time.time())% self.STEP_DELAY_TARGET
        time.sleep(phase_err)
        print "Initial phase error = {}".format(phase_err)
        self.last_step_t = time.time()
        phase_err = \
            (self.phase_target - self.last_step_t + self.STEP_DELAY_TARGET/2.0) \
            % self.STEP_DELAY_TARGET - self.STEP_DELAY_TARGET/2.0
        self.phase_err_i += phase_err
        phase_err_d = (phase_err - self.phase_err) if self.phase_err is not None else 0
        self.phase_err = phase_err
        self.phase_delta = 0.1*(self.phase_err + self.phase_err_i + phase_err_d)

        # observation
        obs_list = []
        for _ in range(self.len_obs):
            while True:
                try:
                    next_states = self.q_obs.get(timeout=0.05)
                    self.q_obs.task_done()
                    break
                except:
                    with self.cnt_q_except.get_lock():
                        if self.cnt_q_except.value>0:
                            self.cnt_q_except.value -= 1
                        else:
                            return None
                        print "[reset()]: exception getting observation. {}.".format(
                            self.cnt_q_except.value)
                    time.sleep(0.1)
            obs_list.append(next_states)
        next_state = self.__compile_obs(obs_list)

        self.is_env_resetting.clear()
        print "[reset()]: done!"
        return next_state

    def __init_queue(self):
        """Initialize queues, unsetting is_q_ready in progress."""
        self.is_q_ready.clear()
        buffer_sizes = self.buffer_sizes
        self.q_obs = Queue(buffer_sizes['obs'])
        self.q_reward = Queue(buffer_sizes['reward'])
        self.q_action = Queue(1)
        self.q_done = Queue(1)
        self.q = {
            'obs': self.q_obs,
            'reward': self.q_reward,
            'action': self.q_action,
            'done': self.q_done}
        self.is_q_ready.set()
        self.is_q_cleared.clear()
        with self.cnt_q_except.get_lock():
            self.cnt_q_except.value = self.MAX_EXCEPTION

    def __queue_monitor(self):
        """Monitors queue exceptions and empty queues necessary.

        This method is loaded as a daemon thread to monitor the status of
        inter-process queues during trainging and empty queues to shutdown the
        underlying PIPE if there are too many exceptions or the simulator
        frontend process is ready for termination (but may still waiting for
        the PIPEs to be emptied).

        For counting exceptions this method periodically check the value of a
        accumulator shared between this and other threads and processes that
        may utilize the queues (e.g. the `__step()` method).
        """
        print "[__queue_monitor]: queue monitor started."
        while True:
            with self.cnt_q_except.get_lock():
                # if termination condition is met
                if self.cnt_q_except.value <= 0 or \
                   self.is_envnode_terminatable.is_set():
                    print ("[__queue_monitor]: num of q exceptions {}, "
                           "term {}.").format(
                               self.cnt_q_except.value,
                               self.is_envnode_terminatable.is_set())

                    # block further access to the queues
                    if self.is_q_ready.is_set():
                        self.is_q_ready.clear()
                        print "[__queue_monitor]: setting is_q_ready {}".format(
                            self.is_q_ready.is_set())

                    # empty queues
                    for n, q in self.q.iteritems():
                        print "[__queue_monitor]: emptying queue {}".format(n)
                        while True:
                            try:
                                q.get(timeout=0.1)
                                q.task_done()
                            except:
                                break
                        if q.qsize()==0:
                            print "[__queue_monitor]: queue {} emptied.".format(n)

                    # set flag only if all queues are cleared
                    if sum([q.qsize() for _, q in self.q.iteritems()])==0:
                        print "[__queue_monitor]: all queues emptied."
                        self.is_q_cleared.set()

            # Return only if envnode is down and queues are properly cleared
            # Note although envnode_up event may be cleared by `reset()`, the
            # q_cleared event can only be set here in this thread.
            if self.is_q_cleared.is_set() and \
               not self.is_envnode_up.is_set():
                break
            else:
                time.sleep(2.0)

        print "[__queue_monitor]: returning..."
        return

    def __node_monitor(self):
        self.is_backend_up.clear()
        self.is_q_cleared.set()
        self.is_envnode_up.clear()    # default to node_down
        self.is_envnode_terminatable.clear()  # prevents q monitor from turning down
        self.is_envnode_resetting.set()
        self.is_receiving_obs.clear()

        while not self.is_exiting.is_set():
            while not self.is_env_resetting.is_set():
                time.sleep(1.0)
                print "[__node_monitor] waiting for reset."
            try:
                # env done check loop
                #   reset() should clear `is_env_done`  
                while self.is_env_done.is_set():
                    print ("[__node_monitor]: env is done, "
                           "waiting for reset.")
                    time.sleep(1.0)

                # start simulation backend
                #   __start_backend() should set `is_backend_up` if successful
                self.__start_backend()
                while not self.is_backend_up.is_set():
                    print "[__node_monitor]: backend not up..."
                    time.sleep(1.0)


                # set up inter-process queues and monitor
                while not self.is_q_cleared.is_set():
                    print "[__node_monitor]: queues not cleared.."
                    time.sleep(1.0)
                self.__init_queue() # should clear `is_q_cleared` and set `is_q_ready`
                self.is_envnode_terminatable.clear()  # prevent queue mon from empty
                thread_queue_monitor = threading.Thread(target=self.__queue_monitor)
                thread_queue_monitor.start()
                print "[__node_monitor]: queue and monitor up and running."


                # run the frontend node
                print "[__node_monitor]: running new env node."
                node = DrivingSimulatorNode(
                    self.q_obs, self.q_reward, self.q_action, self.q_done,
                    self.is_backend_up, self.is_q_ready, self.is_envnode_up,
                    self.is_envnode_terminatable, self.is_envnode_resetting,
                    self.is_receiving_obs,
                    self.defs_obs, self.defs_reward, self.defs_action,
                    self.rate_action, self.is_dummy_action,
                    self.NODE_TIMEOUT)
                node.start()
                node.join()
                # ==== protect from resetting by reset() ===
                # ==== u ntil inside node.run in the next loop
                self.is_envnode_resetting.set()
                # ===========================================
                self.is_envnode_terminatable.clear()
                self.is_envnode_up.clear()
                self.is_receiving_obs.clear()

                # wait for queue mon to return
                thread_queue_monitor.join()  # implies queue cleared and not ready

                # wait for backend to end
                self.__kill_backend()
                self.is_backend_up.clear()
            except Exception as e:
                print "[__node_monitor]: exception running node ({}).".format(
                    e.message)
                traceback.print_exc()
                time.sleep(1.0)
            finally:
                print "[__node_monitor]: finished running node."
                while node.is_alive():
                    print ("[__node_monitor]: process {} termination in"
                           "progress..").format(node.pid)
                    # node.terminate()
                    time.sleep(1.0)
                print "[__node_monitor]: terminiated process {}.".format(node.pid)

    def __kill_backend(self):
        self.is_backend_up.clear()
        if len(self.proc_backend)!=0:
            print '[DrSim] terminating backend processes..'
            for proc in self.proc_backend[::-1]:  # killing in reversed order
                if proc.poll() is None:
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception as e:
                        print "[DrSim] process {} shutdown error: {}".format(
                            proc.pid, e.message)
                while proc.poll() is None:
                    print ("[DrSim] backend process {} termination "
                           " in progress...").format(proc.pid)
                    time.sleep(1.0)
                time.sleep(1.0)
        self.proc_backend = []

    def __start_backend(self):
        print '[DrSim]: initializing backend...'
        self.is_backend_up.clear()
        for i, cmd in enumerate(self.backend_cmds):
            cmd = map(str, cmd)
            proc = subprocess.Popen(cmd)
            self.proc_backend.append(proc)
            print '[DrSim]: cmd [{}] running with PID {}'.format(
                ' '.join(cmd), proc.pid)
            time.sleep(1.0)
        self.is_backend_up.set()
        print '[DrSim]: backend initialized. PID: {}'.format(
            [p.pid for p in self.proc_backend])

    def __import_defs(self, defs):
        """Import class based on package and class names in defs.

        To save environment clients from importing topic message classes,
        this method inspects the class_or_name field of definition tuple.
        If a string is found, we import the class and substitute the
        original name string.

        Examples:
            `std_msgs.msg.Char` -> Char class
        """
        imported_defs = []
        for topic, class_or_name in defs:
            # substitute package_name.class_name with imported class
            if type(class_or_name) is str:
                package_name = '.'.join(class_or_name.split('.')[:-1])
                class_name = class_or_name.split('.')[-1]
                class_or_name = getattr(
                    importlib.import_module(package_name), class_name)
            imported_defs.append((topic, class_or_name))
        return imported_defs

    def exit(self):
        self.is_exiting.set()
        self.is_envnode_terminatable.set()
        while self.is_envnode_up.is_set():
            print "[exit()]: waiting envnode to finish."
            time.sleep(1.0)
        self.__kill_backend()


# the transient part
class DrivingSimulatorNode(multiprocessing.Process):
    def __init__(self,
             q_obs, q_reward, q_action, q_done,
             is_backend_up, is_q_ready, is_envnode_up,
             is_envnode_terminatable, is_envnode_resetting,
             is_receiving_obs,
             defs_obs, defs_reward, defs_action,
             rate_action, is_dummy_action, node_timeout):
        super(DrivingSimulatorNode, self).__init__()

        # inter-thread queues for buffering experience
        self.q_obs = q_obs
        self.q_reward = q_reward
        self.q_action = q_action
        self.q_done = q_done
        self.is_q_ready = is_q_ready
        self.q = {
            'obs': self.q_obs,
            'reward': self.q_reward,
            'action': self.q_action,
            'done': self.q_done}

        # experience definition
        self.defs_obs = defs_obs
        self.defs_reward = defs_reward
        self.defs_action = defs_action

        self.rate_action = rate_action
        self.is_dummy_action = is_dummy_action
        self.Q_TIMEOUT = 1.0
        self.NODE_TIMEOUT = node_timeout

        # inter-thread events
        self.is_backend_up = is_backend_up
        self.is_envnode_up = is_envnode_up
        self.first_time = Event()
        self.is_envnode_terminatable = is_envnode_terminatable
        self.is_envnode_resetting = is_envnode_resetting
        self.is_receiving_obs = is_receiving_obs

    def run(self):
        """Run the simulator backend for one episode.

        This process target method run the simulator backend for one episode.
        It initializes nodes to listen to simulator topics and publish action
        and restart signals to the simulation.

        Handshake procedures:
            1. Setup listener and sender nodes.
            2. Signal simulator restart and wait for new observation
               (with timeout).
            3. Send space, 1, g to /autoDrive_KeyboadMode to start the car.
            4. Action sender doesn't send keys until simulator is up and car is
               started (to avoid flooding signals in step 3).
            5. Heartbeat checker listen to simulator heartbeat (async) and set
               is_envnode_terminatable flag to terminate this process (i.e. poison pill).
        """
        print "[EnvNode]: started frontend process: {}".format(self.name)

        # === Specify experience preparation functions ===
        # TODO: should be able to specify arbitrary ways to prepare exp
        self.list_prep_exp = []
        for i in range(len(self.defs_obs)):
            if i<1:
                self.list_prep_exp.append(self.__prep_image)
            else:
                self.list_prep_exp.append(self.__prep_reward)
        self.list_prep_exp += [self.__prep_reward]*len(self.defs_reward)

        # === Setup sync events ===
        self.is_receiving_obs.clear()
        self.first_time.set()

        # === Initialize frontend ROS node ===
        print "[EnvNode]: initialiting node..."
        rospy.init_node('DrivingSimulatorEnv')
        self.brg = CvBridge()

        # obs + Reward subscribers (synchronized)
        f_subs = lambda defs: message_filters.Subscriber(defs[0], defs[1])
        self.ob_subs = map(f_subs, self.defs_obs)
        self.reward_subs = map(f_subs, self.defs_reward)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.ob_subs + self.reward_subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.__enque_exp)

        # heartbeat subscribers
        rospy.Subscriber('/rl/simulator_heartbeat', Bool, self.__enque_done)
        rospy.Subscriber('/rl/is_running', Bool, self.__heartbeat_checker)

        # action and restart publishers
        f_pubs = lambda defs: rospy.Publisher(
            defs[0], defs[1], queue_size=100, latch=True)
        self.action_pubs = map(f_pubs, self.defs_action)
        # publish action periodically if not using dummy action
        if not self.is_dummy_action:
            self.actor_loop = Timer(
                rospy.Duration(1.0/self.rate_action), self.__take_action)
        self.restart_pub = rospy.Publisher(
            '/rl/simulator_restart', Bool, queue_size=10, latch=True)
        print "[EnvNode]: node initialized."

        # === Simulator initialization: ===
        #   1. signal start
        #   2. wait for new observation and count cnt_fail seconds
        #   3. mark initialization failed and break upon timeout
        #   4. mark initialization failed and break upon termination flag 
        print "[EnvNode]: signal simulator restart"
        self.restart_pub.publish(True)
        cnt_fail = self.NODE_TIMEOUT
        flag_fail = False
        while not self.is_receiving_obs.is_set() or self.first_time.is_set():
            cnt_fail -= 1 if cnt_fail>=0 else 0
            print "[EnvNode]: simulator not up, wait for {} sec(s)...".format(cnt_fail)
            time.sleep(1.0)
            if cnt_fail<=0 or self.is_envnode_terminatable.is_set():
                self.restart_pub.publish(False)
                print "[EnvNode]: simulation initialization failed, "
                flag_fail = True
                break

        # === Run simulator ===
        #   1. set `is_envnode_up` Event
        #   2. Periodically check backend status and termination event
        t = time.time()
        if not flag_fail:
            print "[EnvNode]: simulator up and receiving obs."
            for i in range(2):
                print "[EnvNode]: simulation start in {} secs.".format(i)
                time.sleep(1.0)
            self.is_envnode_resetting.clear()
            self.is_envnode_up.set()
            # Loop check if simulation episode is done
            while self.is_backend_up.is_set() and \
                  not self.is_envnode_terminatable.is_set():
                time.sleep(0.2)
        else:
            # if initialization failed terminate the front end
            self.is_envnode_terminatable.set()

        # shutdown frontend ROS threads
        rospy.signal_shutdown('[DrivingSimulatorEnv]: simulator terminated.')

        # Close queues for this process
        for key in self.q:
            self.q[key].close()

        # Return from this process
        print ("[EnvNode]: returning from run in process: "
               "{} PID: {}, after {:.2f} secs...").format(
                   self.name, self.pid, time.time()-t)
        secs = 2
        while secs != 0:
            print "..in {} secs".format(secs)
            secs -= 1
            time.sleep(1.0)
        print "[EnvNode]: Now!"

        # manually set this event to notify queue monitor to clear queues
        self.is_envnode_terminatable.set()

        return

    def __enque_exp(self, *args):
        # check backend and queue status, return if not ready.
        if not self.is_backend_up.is_set():
            print "[__enque_exp]: backend not up."
            return
        if not self.is_q_ready.is_set():
            # print "[__enque_exp]: queue not ready."
            return

        # process obseravtion and reward
        num_obs = len(self.ob_subs)
        num_reward = len(self.reward_subs)
        args = list(args)
        # observations
        exp = [_func(x) for (_func, x) in zip(self.list_prep_exp, args)]
        try:
            obs = exp[:num_obs] if num_obs>1 else [exp[0]]
            if self.q_obs.full():
                self.q_obs.get_nowait()
                self.q_obs.task_done()
            self.q_obs.put(obs, timeout=self.Q_TIMEOUT)
        except:
            # print "[__enque_exp]: q_obs update exception!"
            pass
        try:
            rewards = exp[num_obs:] if num_reward>1 else [exp[num_obs]]
            if self.q_reward.full():
                self.q_reward.get_nowait()
                self.q_reward.task_done()
            self.q_reward.put(rewards, timeout=self.Q_TIMEOUT)
        except:
            # print "[__enque_exp]: q_reward update exception!"
            pass
        if not self.is_receiving_obs.is_set():
            print "[__enque_exp]: first observation received."
            self.is_receiving_obs.set()  # assume simulator is up after first obs
        # print "[__enque_exp]: {}".format(args[num_obs:])

    def __prep_image(self, img):
         return self.brg.compressed_imgmsg_to_cv2(img, 'rgb8')

    def __prep_reward(self, reward):
        return reward.data

    def __enque_done(self, data):
        # check backend and queue status, return if not ready.
        if not self.is_backend_up.is_set():
            print "[__enque_done]: backend not up."
            return
        if not self.is_q_ready.is_set():
            # print "[__enque_done]: queue not ready."
            return

        done = not data.data
        try:
            if self.q_done.full():
                self.q_done.get_nowait()
                self.q_done.task_done()
            self.q_done.put(done, timeout=self.Q_TIMEOUT)
        except:
            print "__enque_done: q_done full."
            pass
        # print "__eqnue_done: {}".format(done)

    def __take_action(self, data):
        # check backend and queue status, return if not ready.
        if not self.is_backend_up.is_set():
            print "[__take_action]: backend not up."
            return
        if not self.is_q_ready.is_set():
            # print "[__take_action]: queue not ready."
            return
        if self.first_time.is_set():
            # print "[__take_action]: simulator is not running."
            return

        try:
            actions = self.q_action.get_nowait()
            self.q_action.put_nowait(actions)
            self.q_action.task_done()
        except:
            # print "[__take_action]: get action from queue failed."
            return

        if self.is_receiving_obs.is_set():
            # print "__take_action: {}, q len {}".format(
            #     actions, self.q_action.qsize()
            # )
            for i in range(len(self.action_pubs)):
                self.action_pubs[i].publish(actions[i])
        else:
            # print "[__take_action]: simulator up ({}).".format(
            #      self.is_receiving_obs.is_set())
            pass

    def __heartbeat_checker(self, data):
        print "Heartbeat signal: {}, First time: {}".format(
            data, self.first_time.is_set()
        )
        if not data.data and not self.first_time.is_set():
            self.is_envnode_terminatable.set()
        self.first_time.clear()

