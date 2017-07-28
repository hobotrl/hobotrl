# -*- coding: utf-8 -*-
"""Gym-like environment wrapper for Hobot Driving Simulator.

File name: ros_environment.py
Author: Jingchu Liu
Last Modified: July 27, 2017
"""

# Basic python
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
                 step_delay_target=None):
        """Initialization."""
        self.defs_obs = defs_obs
        self.__compile_obs = lambda *args: func_compile_obs(*args)
        self.len_obs = window_sizes['obs']
        self.defs_reward = defs_reward
        self.__compile_reward = lambda *args: func_compile_reward(*args)
        self.len_reward = window_sizes['reward']
        self.defs_action = defs_action
        self.rate_action = rate_action
        self.buffer_sizes = buffer_sizes
        self.MAX_EXCEPTION = 10
        self.STEP_DELAY_TARGET = step_delay_target
        self.step_delay_ema = 0.0
        self.last_step_t = time.time()

        # initialize queue
        self.is_backend_up = Event()
        self.is_q_ready = Event()  # whether queues are ready
        self.is_node_up = Event()  # whether env node is up
        self.is_node_up.clear()    # default to node_down
        self.__init_queue()

        self.backend_cmds = [
            # reward function
            ['python', '/home/lewis/Projects/hobotrl/playground/initialD/gazebo_rl_reward.py'],
            # simulator backend [Recommend start separately]
            # ['python', '/home/lewis/Projects/hobotrl/playground/initialD/rviz_restart.py']
        ]
        self.proc_backend = []

        # monitor threads 
        self.thread_node_monitor = threading.Thread(target=self.__node_monitor)
        self.thread_node_monitor.start()
        self.thread_queue_monitor = threading.Thread(target=self.__queue_monitor)
        self.thread_queue_monitor.start()

    def step(self, action):
        """Evolve environment by one step.

        Interacts with the backend simulator through multiple queues. Put
        the latest action into queue and extract observations, rewards, and
        done signal from queues.

        Catch exceptions when interacting with queues: 1) modify queue counter for
        queue monitor to shut node down if there are too many exceptions. 2)
        return None for the unsuccessful interactions. The queue counter is
        decreased by 1 per exception and increased by 1 per sucessful step.
        """
        # wait until backend is up
        while True:
            if self.is_node_up.is_set() and \
               self.is_q_ready.is_set() and \
               self.is_backend_up.is_set():
                break
            else:
                # print "[step()]: backend {}, node {}, queue {}.".format(
                #    self.is_backend_up.is_set(),
                #    self.is_node_up.is_set(),
                #    self.is_q_ready.is_set())
                time.sleep(0.1)

        # step delay regularization
        delay = time.time() - self.last_step_t
        self.step_delay_ema += 0.4*(delay - self.step_delay_ema)
        if self.STEP_DELAY_TARGET is not None:
            if delay < self.STEP_DELAY_TARGET:
                time.sleep(self.STEP_DELAY_TARGET-delay)
            else:
                print ("[step()]: delay {:.3f} >= target (:.3f), if happen "
                       "regularly please conconsider increasing target.").format(
                           delay, self.STEP_DELAY_TARGET)
        self.last_step_t = time.time()

        # action
        while True:
            try:
                if self.q_action.full():
                    self.q_action.get(False)
                    self.q_action.task_done()
                self.q_action.put_nowait(action)
                break
            except:
                with self.cnt_q_except.get_lock():
                    self.cnt_q_except.value -= 1
                    print "[step()]: putting action into queue, exception counter {}.".format(
                        self.cnt_q_except.value)
                    time.sleep(0.1)

        # observation
        obs_list = []
        for _ in range(self.len_obs):
            while True:
                try:
                    next_states = self.q_obs.get_nowait()
                    self.q_obs.task_done()
                    break
                except:
                    with self.cnt_q_except.get_lock():
                        self.cnt_q_except.value -= 1
                        print "[step()]: exception getting observation, exception counter {}.".format(
                            self.cnt_q_except.value)
                    time.sleep(0.1)
            obs_list.append(next_states)
        next_state = self.__compile_obs(obs_list)

        # reward
        reward_list = []
        for _ in range(self.len_reward):
            while True:
                try:
                    rewards = self.q_reward.get_nowait()
                    self.q_reward.task_done()
                    break
                except:
                    with self.cnt_q_except.get_lock():
                        self.cnt_q_except.value -= 1
                        print "[step()]: exception getting reward, exception counter {}.".format(
                            self.cnt_q_except.value)
                    time.sleep(0.1)
            reward_list.append(rewards)
        print "[step()]: reward vector {}".format(reward_list)
        reward = self.__compile_reward(reward_list)

        # done
        while True:
            try:
                done = self.q_done.get_nowait()
                self.q_done.task_done()
                break
            except:
                with self.cnt_q_except.get_lock():
                    self.cnt_q_except.value -= 0
                print ("[step()]: exception getting done, "
                       "counter {}.").format(self.cnt_q_except.value)
                time.sleep(0.1)

        # info
        info = None

        # increase counter by 1 per sucessfuly step
        with self.cnt_q_except.get_lock():
            if self.cnt_q_except.value<self.MAX_EXCEPTION:
                self.cnt_q_except.value += 1

        print "[step()]: action {}, reward {}, done {}.".format(
            action, reward, done)
        return next_state, reward, done, info

    def reset(self):
        # wait until backend is up
        while True:
            if self.is_node_up.is_set() and \
               self.is_q_ready.is_set() and \
               self.is_backend_up.is_set():
                break
            else:
                print "[reset]: backend {}, node {}, queue {}.".format(
                    self.is_backend_up.is_set(),
                    self.is_node_up.is_set(),
                    self.is_q_ready.is_set())
                time.sleep(0.1)
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
        self.cnt_q_except = Value('i', self.MAX_EXCEPTION)
        self.is_q_ready.set()

    def __queue_monitor(self):
        """Periodically check queue status and signal queue down if there are
        too many queue exceptions."""
        print "[__queue_monitor]: queue monitor started."
        while True:
            with self.cnt_q_except.get_lock():
                if self.cnt_q_except.value<=0:
                    print "[__queue_monitor]: num of queue exceptions exceeded limit."
                    if self.is_q_ready.is_set():
                        self.is_q_ready.clear()
                        print "[__queue_monitor]: setting is_q_ready {}".format(
                            self.is_q_ready.is_set())
            time.sleep(5.0)

    def __node_monitor(self):
        while True:
            try:
                if not self.is_node_up.is_set():
                    self.__start_backend()
                while not self.is_backend_up.is_set():
                    print "[__node_monitor]: backend not up..."
                    time.sleep(1.0)
                print "[__node_monitor]: running new node."
                self.__init_queue()
                print "[__node_monitor]: set up new queue."
                node = DrivingSimulatorNode(
                    self.q_obs, self.q_reward, self.q_action, self.q_done,
                    self.is_backend_up, self.is_q_ready, self.is_node_up,
                    self.defs_obs, self.defs_reward, self.defs_action,
                    self.rate_action
                )
                node.start()
                node.join()
            except Exception as e:
                print "[__node_monitor]: exception running node ({}).".format(
                    e.message)
                time.sleep(1.0)
            finally:
                print "[__node_monitor]: finished running node."
                while True:
                    node.terminate()
                    time.sleep(1.0)
                    if node.is_alive():
                        print ("[__node_monitor]: process {} termination in"
                               "progress..").format(node.pid)
                        continue
                    else:
                        break
                print "[__node_monitor]: terminiated process {}.".format(node.pid)

    def __kill_backend(self):
        self.is_backend_up.clear()
        if len(self.proc_backend)!=0:
            print '[DrSim] terminating backend processes..'
            for proc in self.proc_backend:
                if proc.poll() is None:
                    proc.terminate()
                while proc.poll() is None:
                    print ("[DrSim] backend process {} termination "
                           " in progress...").format(proc.pid)
                    time.sleep(1.0)
        self.proc_backend = []

    def __start_backend(self):
        print '[DrSim]: initializing backend...'
        self.is_backend_up.clear()
        self.proc_backend += [subprocess.Popen(cmd) for cmd in self.backend_cmds]
        self.is_backend_up.set()
        print '[DrSim]: backend initialized.'


class DrivingSimulatorNode(multiprocessing.Process):
    def __init__(self,
             q_obs, q_reward, q_action, q_done,
             is_backend_up, is_q_ready, is_node_up,
             defs_obs, defs_reward, defs_action,
             rate_action):
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
        self.Q_TIMEOUT = 1.0

        self.is_backend_up = is_backend_up
        self.is_node_up = is_node_up
        self.first_time = Event()
        self.terminatable = Event()
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
               terminatable flag to terminate this process (i.e. poison pill).
        """
        print "[EnvNode]: started process: {}".format(self.name)
        self.list_prep_exp = [self.__prep_image] + \
                [self.__prep_reward]*len(self.defs_reward)
        self.is_node_up.clear()
        self.first_time.set()
        self.terminatable.clear()
        self.is_receiving_obs.clear()
        self.car_started.clear()

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
        self.actor_loop = Timer(
            rospy.Duration(1.0/self.rate_action), self.__take_action)
        self.restart_pub = rospy.Publisher(
            '/rl/simulator_restart', Bool, queue_size=10, latch=True)
        print "[EnvNode]: node initialized."

        # Simulator initialization
        #   Wait new_obs for cnt_fail seconds  
        #   send ' ', 'g', '1' until new obs is observed
        print "[EnvNode]: starting simulator."
        print "[EnvNode]: signal simulator restart"
        self.restart_pub.publish(True)
        cnt_fail = 15
        flag_fail = False  # initialization failed flag
        while not self.is_receiving_obs.is_set() and not flag_fail:
            cnt_fail -= 1
            print "[EnvNode]: simulator not up, wait for {} sec(s)...".format(cnt_fail)
            time.sleep(1.0)
            if cnt_fail==0:
                self.restart_pub.publish(False)
                print "[EnvNode]: simulation initialization failed, "
                flag_fail = True

        # Simulator run
        t = time.time()
        if not flag_fail:
            __thread_start_car = threading.Thread(target=self.__start_car)
            __thread_start_car.start()
            # Loop check if simulation episode is done
            while self.is_backend_up.is_set() and not self.terminatable.is_set():
                time.sleep(0.2)
        else:
            pass

        # rospy.signal_shutdown('[DrivingSimulatorEnv]: simulator terminated.')

        # Close queues for this process
        for key in self.q:
            self.q[key].close()

        print "Returning from run in process: {} PID: {}, after {:.2f} secs...".format(
            self.name, self.pid, time.time()-t)
        secs = 3
        while secs != 0:
            print "..in {} secs".format(secs)
            secs -= 1
            time.sleep(1.0)
        return

    def __start_car(self):
        time.sleep(1.0)
        print "[EnvNode]: sending key 'space' ..."
        self.action_pubs[0].publish(ord(' '))
        for _ in range(2):
            time.sleep(0.5)
            print "[EnvNode]: sending key '1' ..."
            self.action_pubs[0].publish(ord('1'))
            time.sleep(0.5)
            print "[EnvNode]: sending key 'g' ..."
            self.action_pubs[0].publish(ord('g'))
        self.car_started.set()
        self.is_node_up.set()

    def __enque_exp(self, *args):
        # check backend and queue status, return if not ready.
        if not self.is_backend_up.is_set():
            print "[__enque_exp]: backend not up."
            return
        if not self.is_q_ready.is_set():
            print "[__enque_exp]: queue not ready."
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
        # print "__enque_exp: {}".format(args[num_obs:])
        self.is_receiving_obs.set()  # assume simulator is up after first obs

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
            print "[__enque_done]: queue not ready."
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
            print "[__take_action]: queue not ready."
            return

        try:
            actions = self.q_action.get_nowait()
            self.q_action.put_nowait(actions)
            self.q_action.task_done()
        except:
            print "[__take_action]: get action from queue failed."
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
            self.terminatable.set()
        self.first_time.clear()


