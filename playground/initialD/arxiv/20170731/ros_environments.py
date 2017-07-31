# -*- coding: utf-8 -*-
"""Gym-like environment wrapper for Hobot Driving Simulator.

File name: ros_environment.py
Author: Jingchu Liu
Last Modified: July 27, 2017
"""

# Basic python
import signal
import time
# Threading and Multiprocessing
import threading
import subprocess
import multiprocessing
from multiprocessing import JoinableQueue as Queue
from multiprocessing import Value, Event, Pipe
# Image
from scipy.misc import imresize
import cv2
from cv_bridge import CvBridge, CvBridgeError
# ROS
import rospy
from rospy.timer import Timer
import message_filters
from std_msgs.msg import Char, Bool, Float32
from sensor_msgs.msg import Image


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
    """
    def __init__(self,
                 defs_obs, func_compile_reward,
                 defs_reward, func_compile_obs,
                 defs_action, rate_action,
                 window_sizes, buffer_sizes,
                 step_delay_target=None,
                 is_dummy_action=False):
        """Initialization."""
        # params
        self.defs_obs = defs_obs
        self.__compile_obs = lambda *args: func_compile_obs(*args)
        self.len_obs = window_sizes['obs']

        self.defs_reward = defs_reward
        self.__compile_reward = lambda *args: func_compile_reward(*args)
        self.len_reward = window_sizes['reward']

        self.defs_action = defs_action
        self.rate_action = rate_action
        self.is_dummy_action = is_dummy_action

        self.buffer_sizes = buffer_sizes
        self.MAX_EXCEPTION = 10  # maximum number of q exceptions

        self.STEP_DELAY_TARGET = step_delay_target  # target delay for each step()
        self.last_step_t = time.time()

        # inter-process flags (mp.Event)
        self.is_backend_up = Event()
        self.is_q_ready = Event()  # whether qs are ready
        self.is_q_cleared = Event()  # whether qs from prevoius session is cleared
        self.is_envnode_up = Event()  # whether env node is up
        self.is_envnode_terminatable = Event()
        self.is_env_done = Event()

        # backend specs
        self.backend_cmds = [
            # roscore
            # ['roscore'],
            # reward function
            # ['python', '/home/lewis/Projects/hobotrl/playground/initialD/gazebo_rl_reward.py'],
            # simulator backend [Recommend start separately]
            # ['python', '/home/lewis/Projects/hobotrl/playground/initialD/rviz_restart.py']
        ]
        self.proc_backend = []

        # monitor threads 
        self.thread_node_monitor = threading.Thread(target=self.__node_monitor)
        self.thread_node_monitor.start()

    def step(self, action):
        # step delay regularization
        delay = time.time() - self.last_step_t
        if self.STEP_DELAY_TARGET is not None:
            if delay < self.STEP_DELAY_TARGET:
                time.sleep(self.STEP_DELAY_TARGET-delay)
            else:
                print ("[step()]: delay {:.3f} >= target {:.3f}, if happen "
                       "regularly please conconsider increasing target.").format(
                           delay, self.STEP_DELAY_TARGET)
        self.last_step_t = time.time()

        # do step
        while True:
            ret = self.__step(action)
            if ret is not None:
                break
            time.sleep(1.0)
        next_state, reward, done, info = ret
        print "[step()]: action {}, reward {}, done {}.".format(
            action, reward, done)

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
        """
        # wait until envnode, q, and backend is up and running
        while True:
            if self.is_envnode_up.is_set() and \
               self.is_q_ready.is_set() and \
               self.is_backend_up.is_set():
                break
            else:
                print "[__step()]: backend {}, node {}, queue {}.".format(
                   self.is_backend_up.is_set(),
                   self.is_envnode_up.is_set(),
                   self.is_q_ready.is_set())
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
                        print "[__step()]: exception getting observation. {}.".format(
                            self.cnt_q_except.value)
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
                        print "[__step()]: exception getting reward. {}.".format(
                            self.cnt_q_except.value)
                    time.sleep(0.1)
            reward_list.append(rewards)
        print "[step()]: reward vector {}".format(reward_list)
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
        info = None

        # increase counter by `credit` per successfuly step
        credit = 1 + self.len_obs + self.len_reward + 1
        with self.cnt_q_except.get_lock():
            while credit>0 and self.cnt_q_except.value<self.MAX_EXCEPTION:
                self.cnt_q_except.value += 1
                credit -= 1

        return next_state, reward, done, info

    def reset(self):
        # wait until backend is up
        while True:
            if self.is_envnode_up.is_set() and \
               self.is_q_ready.is_set() and \
               self.is_backend_up.is_set():
                break
            else:
                print "[reset]: backend {}, node {}, queue {}.".format(
                    self.is_backend_up.is_set(),
                    self.is_envnode_up.is_set(),
                    self.is_q_ready.is_set())
                time.sleep(0.5)
        states = self.q_obs.get()[0]
        self.q_obs.task_done()
        state = self.__compile_obs(states)
        return state

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
        self.cnt_q_except = Value('i', self.MAX_EXCEPTION)
        self.is_q_ready.set()
        self.is_q_cleared.clear()

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

            # return only if envnode is down and queues are cleared
            if self.is_q_cleared.is_set() and \
               not self.is_envnode_up.is_set():
                break
            else:
                time.sleep(2.0)

        print "[__queue_monitor]: returning..."
        return

    def __node_monitor(self):
        self.is_q_cleared.set()
        self.is_envnode_up.clear()    # default to node_down
        while True:
            try:
                # start simulation backend
                self.__start_backend()
                while not self.is_backend_up.is_set():
                    print "[__node_monitor]: backend not up..."
                    time.sleep(1.0)

                # set up inter-process queues
                while not self.is_q_cleared.is_set():
                    print "[__node_monitor]: queues not cleared.."
                    time.sleep(1.0)
                self.is_envnode_terminatable.clear()  # prevents q monitor from turning down
                self.__init_queue()
                thread_queue_monitor = threading.Thread(target=self.__queue_monitor)
                thread_queue_monitor.start()
                print "[__node_monitor]: set up new queue."

                # run the frontend node
                print "[__node_monitor]: running new env node."
                node = DrivingSimulatorNode(
                    self.q_obs, self.q_reward, self.q_action, self.q_done,
                    self.is_backend_up, self.is_q_ready, self.is_envnode_up,
                    self.is_envnode_terminatable,
                    self.defs_obs, self.defs_reward, self.defs_action,
                    self.rate_action, self.is_dummy_action)
                node.start()
                node.join()
                self.is_envnode_up.clear()

                self.__kill_backend()
                thread_queue_monitor.join()
            except Exception as e:
                print "[__node_monitor]: exception running node ({}).".format(
                    e.message)
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
        self.proc_backend = []

    def __start_backend(self):
        print '[DrSim]: initializing backend...'
        self.is_backend_up.clear()
        for cmd in self.backend_cmds:
            proc = subprocess.Popen(cmd)
            self.proc_backend.append(proc)
            print '[DrSim]: cmd [{}] running with PID {}'.format(
                ' '.join(cmd), proc.pid)
            time.sleep(1.0)
        self.is_backend_up.set()
        print '[DrSim]: backend initialized. PID: {}'.format(
            [p.pid for p in self.proc_backend])


class DrivingSimulatorNode(multiprocessing.Process):
    def __init__(self,
             q_obs, q_reward, q_action, q_done,
             is_backend_up, is_q_ready, is_envnode_up,
             is_envnode_terminatable,
             defs_obs, defs_reward, defs_action,
             rate_action, is_dummy_action):
        super(DrivingSimulatorNode, self).__init__()

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

        self.defs_obs = defs_obs
        self.defs_reward = defs_reward
        self.defs_action = defs_action

        self.rate_action = rate_action
        self.is_dummy_action = is_dummy_action
        self.Q_TIMEOUT = 1.0

        self.is_backend_up = is_backend_up
        self.is_envnode_up = is_envnode_up
        self.first_time = Event()
        self.is_envnode_terminatable = is_envnode_terminatable
        self.is_receiving_obs = Event()
        self.car_started = Event()

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
        self.list_prep_exp = [self.__prep_image] + \
                [self.__prep_reward]*len(self.defs_reward)
        # setup flags
        self.is_envnode_up.clear()
        self.is_receiving_obs.clear()
        self.car_started.clear()
        self.first_time.set()
        self.is_envnode_terminatable.clear()

        while not self.is_backend_up.is_set():
            print "[EnvNode]: waiting for backend..."
            time.sleep(1.0)

        # Initialize ROS node
        print "[EnvNode]: initialiting node..."
        rospy.init_node('DrivingSimulatorEnv')
        self.brg = CvBridge()
        # Obs + Reward: synced
        f_subs = lambda defs: message_filters.Subscriber(defs[0], defs[1])
        self.ob_subs = map(f_subs, self.defs_obs)
        self.reward_subs = map(f_subs, self.defs_reward)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.ob_subs + self.reward_subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.__enque_exp)
        # Heartbeat
        rospy.Subscriber('/rl/simulator_heartbeat', Bool, self.__enque_done)
        rospy.Subscriber('/rl/is_running', Bool, self.__heartbeat_checker)
        f_pubs = lambda defs: rospy.Publisher(
            defs[0], defs[1], queue_size=100, latch=True)
        self.action_pubs = map(f_pubs, self.defs_action)
        if not self.is_dummy_action:
            self.actor_loop = Timer(
                rospy.Duration(1.0/self.rate_action), self.__take_action)
        self.restart_pub = rospy.Publisher(
            '/rl/simulator_restart', Bool, queue_size=10, latch=True)
        print "[EnvNode]: node initialized."

        # Simulator initialization
        #   1. wait for new observation and count cnt_fail seconds
        #   2. mark initialization failed and break upon timeout
        #   3. mark initialization failed and break upon termination flag 
        print "[EnvNode]: starting simulator."
        print "[EnvNode]: signal simulator restart"
        self.restart_pub.publish(True)
        cnt_fail = 15
        flag_fail = False  # initialization failed flag
        while not self.is_receiving_obs.is_set():
            cnt_fail -= 1 if cnt_fail>=0 else 0
            print "[EnvNode]: simulator not up, wait for {} sec(s)...".format(cnt_fail)
            time.sleep(1.0)
            if cnt_fail<=0 or self.is_envnode_terminatable.is_set():
                self.restart_pub.publish(False)
                print "[EnvNode]: simulation initialization failed, "
                flag_fail = True
                break

        # Simulator run
        t = time.time()
        if not flag_fail:
            print "[EnvNode]: simulator up and receiving obs."
            # send ' ', 'g', '1' until new obs is observed
            __thread_start_car = threading.Thread(target=self.__start_car)
            __thread_start_car.start()
            __thread_start_car.join()
            self.is_envnode_up.set()
            # Loop check if simulation episode is done
            while self.is_backend_up.is_set() and \
                  not self.is_envnode_terminatable.is_set():
                time.sleep(0.2)
        else:
            pass

        # rospy is shutdown at Ctrl-C, but will it exit on process/thread end?
        rospy.signal_shutdown('[DrivingSimulatorEnv]: simulator terminated.')

        # Close queues for this process
        for key in self.q:
            self.q[key].close()

        print ("[EnvNode]: returning from run in process: "
               "{} PID: {}, after {:.2f} secs...").format(
                   self.name, self.pid, time.time()-t)
        secs = 5
        while secs != 0:
            print "..in {} secs".format(secs)
            secs -= 1
            time.sleep(1.0)
        print "[EnvNode]: Now!"

        self.is_envnode_terminatable.set()

        return

    def __start_car(self):
        time.sleep(1.0)
        print "[EnvNode]: sending key 'space' ..."
        self.action_pubs[0].publish(ord(' '))
        for _ in range(2):
            time.sleep(0.5)
            if self.is_dummy_action:
                print "[EnvNode]: sending key '0' ..."
                self.action_pubs[0].publish(ord('0'))
            else:
                print "[EnvNode]: sending key '1' ..."
                self.action_pubs[0].publish(ord('1'))
            time.sleep(0.5)
            print "[EnvNode]: sending key 'g' ..."
            self.action_pubs[0].publish(ord('g'))
        self.car_started.set()
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
            print "[__enque_exp]: q_obs update exception!"
            pass
        try:
            rewards = exp[num_obs:] if num_reward>1 else [exp[num_obs]]
            if self.q_reward.full():
                self.q_reward.get_nowait()
                self.q_reward.task_done()
            self.q_reward.put(rewards, timeout=self.Q_TIMEOUT)
        except:
            print "[__enque_exp]: q_reward update exception!"
            pass
        if not self.is_receiving_obs.is_set():
            print "[__enque_exp]: first observation received."
            self.is_receiving_obs.set()  # assume simulator is up after first obs
        # print "[__enque_exp]: {}".format(args[num_obs:])

    def __prep_image(self, img):
         return imresize(
             self.brg.imgmsg_to_cv2(img, 'rgb8'),
             (640, 640))

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

        try:
            actions = self.q_action.get_nowait()
            self.q_action.put_nowait(actions)
            self.q_action.task_done()
        except:
            # print "[__take_action]: get action from queue failed."
            return

        if self.is_receiving_obs.is_set() and self.car_started.is_set():
            # print "__take_action: {}, q len {}".format(
            #     actions, self.q_action.qsize()
            # )
            map(
                lambda args: args[0].publish(args[1]),
                zip(self.action_pubs, actions)
            )
        else:
            print "[__take_action]: simulator up ({}), car started ({})".format(
                 self.is_receiving_obs.is_set(), self.car_started.is_set())
            pass

    def __heartbeat_checker(self, data):
        print "Heartbeat signal: {}, First time: {}".format(
            data, self.first_time.is_set()
        )
        if not data.data and not self.first_time.is_set():
            self.is_envnode_terminatable.set()
        self.first_time.clear()


