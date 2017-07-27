# -*- coding: utf-8 -*-
# Basic python
import time
# Threading
import threading
# Multi-process
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
    def __init__(self, defs_obs, defs_reward, defs_action,
                 rate_action, buffer_sizes):
        """Initialization.
        :param topics_obs:
        :param topics_reward:
        :param topics_action:
        :param rate_action:
        :param buffer_sizes:
        """
        self.buffer_sizes = buffer_sizes
        self.q_ready = Event()
        self.emergency_shutdown = Event()
        self.node_up = Event()
        self.node_up.clear()
        self.__init_queue()

        # pub and sub definitions
        self.defs_obs = defs_obs
        self.defs_reward = defs_reward
        self.defs_action = defs_action
        self.rate_action = rate_action

        # monitor threads 
        self.thread_node_monitor = threading.Thread(target=self.node_monitor)
        self.thread_node_monitor.start()
        self.thread_queue_monitor = threading.Thread(target=self.queue_monitor)
        self.thread_queue_monitor.start()

    def __init_queue(self):
        self.q_ready.clear()
        buffer_sizes = self.buffer_sizes
        self.q_obs = Queue(buffer_sizes['obs'])
        self.q_reward = Queue(buffer_sizes['reward'])
        self.q_action = Queue(buffer_sizes['action'])
        self.q_done = Queue(1)
        self.cnt_q_except = Value('i', 30)
        self.q_ready.set()

    def queue_monitor(self):
        print "[queue_monitor]: queue monitor started."
        while True:
            with self.cnt_q_except.get_lock():
                if self.cnt_q_except.value<=0:
                    print "[queue_monitor]: queue exceptions exeed limit."
                    self.node_up.clear()
            time.sleep(10.0)

    def node_monitor(self):
        while True:
            try:
                print "[node_monitor]: running new node."
                self.__init_queue()
                print "[node_monitor]: set up new queue."
                node = DrivingSimulatorNode(
                    self.q_obs, self.q_reward, self.q_action, self.q_done,
                    self.q_ready, self.emergency_shutdown, self.node_up,
                    self.defs_obs, self.defs_reward, self.defs_action,
                    self.rate_action
                )
                node.start()
                node.join()
            except:
                pass
                print "[node_monitor]: exception running node."
                time.sleep(1.0)
            finally:
                print "[node_monitor]: finished running node."
                while True:
                    node.terminate()
                    time.sleep(1.0)
                    if node.is_alive():
                        print ("[node_monitor]: process {} termination in"
                               "progress..").format(node.pid)
                        continue
                    else:
                        break
                print "[node_monitor]: terminiated process {}.".format(node.pid)

    def step(self, action):
        while not self.node_up.is_set() or not self.q_ready.is_set():
            print "[step()]: queue not ready..."
            time.sleep(1.0)
        # === enqueue action ===
        try:
            if self.q_action.full():
                self.q_action.get(False)
                self.q_action.task_done()
            with self.cnt_q_except.get_lock():
                if self.cnt_q_except.value<30:
                    self.cnt_q_except.value += 1
        except:
            with self.cnt_q_except.get_lock():
                self.cnt_q_except.value -= 1
            print ("[step()]: exception emptying action queue, "
                   "counter {}.").format(self.cnt_q_except.value)
            return None, None, None, None
        try:
            self.q_action.put_nowait(action)
            with self.cnt_q_except.get_lock():
                if self.cnt_q_except.value<30:
                    self.cnt_q_except.value += 1
        except:
            with self.cnt_q_except.get_lock():
                self.cnt_q_except.value -= 1
            print ("[step()]: exception putting action into queue, "
                   "counter {}.").format(self.cnt_q_except.value)
            return None, None, None, None
        print "[step()]: action: {}, queue size: {}".format(
            action, self.q_action.qsize())

        # === compile observation ===
        try:
            next_state = self.q_obs.get_nowait()[0]
            self.q_obs.task_done()
            with self.cnt_q_except.get_lock():
                if self.cnt_q_except.value<30:
                    self.cnt_q_except.value += 1
        except:
            with self.cnt_q_except.get_lock():
                self.cnt_q_except.value -= 1
            print ("[step()]: exception getting observation, "
                   "counter {}.").format(self.cnt_q_except.value)
            time.sleep(1.0)
            return None, None, None, None

        # === calculate reward ===
        try:
            rewards = self.q_reward.get_nowait()
            self.q_reward.task_done()
            with self.cnt_q_except.get_lock():
                if self.cnt_q_except.value<30:
                    self.cnt_q_except.value += 1
        except:
            with self.cnt_q_except.get_lock():
                self.cnt_q_except.value -= 1
            print ("[step()]: exception getting reward, "
                   "counter {}.").format(self.cnt_q_except.value)
            time.sleep(1.0)
            return next_state, None, None, None
        print "[step()]: rewards {}".format(rewards)
        reward = -100.0 * float(rewards[0]) + \
                 -10.0 * float(rewards[1]) + \
                10.0 * float(rewards[2]) + \
                 -100.0 * (1 - float(rewards[3]))
        # decide if episode is done
        try:
            done = self.q_done.get_nowait()
            self.q_done.task_done()
            with self.cnt_q_except.get_lock():
                if self.cnt_q_except.value<30:
                    self.cnt_q_except.value += 1
        except:
            with self.cnt_q_except.get_lock():
                self.cnt_q_except.value -= 1
            print ("[step()]: exception getting done, "
                   "counter {}.").format(self.cnt_q_except.value)
            time.sleep(1.0)
            return next_state, reward, None, None
        # info
        info = None

        print "[step()]: reward {}, done {}.".format(reward, done)
        return next_state, reward, done, info

    def reset(self):
        while not self.node_up.is_set() or not self.q_ready.is_set():
            print "[reset()]: queue not ready..."
            time.sleep(1.0)
        state = self.q_obs.get()[0]
        self.q_obs.task_done()
        return state


class DrivingSimulatorNode(multiprocessing.Process):
    def __init__(self, q_obs, q_reward, q_action, q_done,
                 q_ready, emergency_shutdown, node_up,
                 defs_obs, defs_reward, defs_action,
                 rate_action):
        super(DrivingSimulatorNode, self).__init__()

        self.q_obs = q_obs
        self.q_reward = q_reward
        self.q_action = q_action
        self.q_done = q_done
        self.q_ready = q_ready
        self.emergency_shutdown = emergency_shutdown
        self.node_up = node_up
        self.q = {'obs': self.q_obs,
                  'reward': self.q_reward,
                  'action': self.q_action,
                  'done': self.q_done}

        self.defs_obs = defs_obs
        self.defs_reward = defs_reward
        self.defs_action = defs_action

        self.rate_action = rate_action

        self.first_time = True
        self.terminatable = False
        self.is_simulator_up = False
        self.car_started = False

    def run(self):
        print "[DrivingSimulatorEnv]: started process: {}".format(self.name)
        self.node_up.clear()

        # Initialize ROS node
        print "[DrivingSimulatorEnv]: initialiting node."
        rospy.init_node('DrivingSimulatorEnv')
        self.brg = CvBridge()
        # === Subscribers ===
        # Obs + Reward: synced
        f_subs = lambda defs: message_filters.Subscriber(defs[0], defs[1])
        self.ob_subs = map(f_subs, self.defs_obs)
        self.reward_subs = map(f_subs, self.defs_reward)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.ob_subs + self.reward_subs, 10, 0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.__enque_exp)
        # Heartbeat
        rospy.Subscriber('/rl/simulator_heartbeat', Bool, self.__enque_done)
        rospy.Subscriber('/rl/is_running', Bool, self.__heartbeat_checker)
        # === Publishers ===
        f_pubs = lambda defs: rospy.Publisher(
            defs[0], defs[1], queue_size=100, latch=True
        )
        self.action_pubs = map(f_pubs, self.defs_action)
        self.actor_loop = Timer(
            rospy.Duration(1.0/self.rate_action), self.__take_action
        )
        self.restart_pub = rospy.Publisher(
            '/rl/simulator_restart', Bool, queue_size=10, latch=True
        )

        # Simulator initialization
        print "[DrivingSimulatorEnv]: signaling simulator restart"
        self.restart_pub.publish(True)

        # Wait new_obs for N_wait seconds  
        # send ' ', 'g', '1' until new obs is observed
        N_wait = 10
        flag_term = False
        while not self.is_simulator_up and not flag_term:
            print "Simulator not up, wait {}...".format(N_wait)
            time.sleep(1.0)
            N_wait -= 1
            if N_wait==0:
                self.restart_pub.publish(False)
                print "Simulation initialization failed, "
                flag_term = True

        t = time.time()

        if not flag_term:
            time.sleep(2.0)
            self.action_pubs[0].publish(ord(' '))
            for _ in range(5):
                time.sleep(0.5)
                self.action_pubs[0].publish(ord('1'))
                time.sleep(0.5)
                self.action_pubs[0].publish(ord('g'))
            self.car_started = True
            while not self.is_simulator_up:
                time.sleep(0.2)
            self.node_up.set()
            # Loop check if simulation episode is done
            while not self.terminatable:
                time.sleep(0.2)
        else:
            pass

        # Close queues for this process
        self.q_obs.close()
        self.q_reward.close()
        self.q_action.close()
        self.q_done.close()

        rospy.signal_shutdown('[DrivingSimulatorEnv]: simulator terminated.')
        if self.emergency_shutdown.is_set():
            self.emergency_shutdown.clear()
        print "Returning from run in process: {} PID: {}, after {:.2f} secs...".format(
            self.name, self.pid, time.time()-t)
        secs = 3
        while secs != 0:
            print "..in {} secs".format(secs)
            secs -= 1
            time.sleep(1.0)
        return

    def __enque_exp(self, *args):
        if not self.q_ready.is_set():
            print "[__enque_exp]: queue not ready."
            return
        # print "__enque_exp: observation received..."
        num_obs = len(self.ob_subs)
        num_reward = len(self.reward_subs)
        args = list(args)
        args[0] = imresize(
            self.brg.imgmsg_to_cv2(args[0], 'rgb8'),
            (640, 640)
        )
        try:
            self.q_obs.put((args[:num_obs]), timeout=0.1)
        except:
            # print "__enque_exp: q_obs full!"
            pass
        try:
            self.q_reward.put(
                (map(lambda data: data.data, args[num_obs:])),
                timeout=0.1
            )
        except:
            # print "__enque_exp: q_reward full!"
            pass
        # print "__enque_exp: {}".format(args[num_obs:])
        self.is_simulator_up = True  # assume simulator is up after first obs

    def __take_action(self, data):
        if not self.q_ready.is_set():
            print "[__enque_exp]: queue not ready."
            return

        try:
            actions = self.q_action.get_nowait()
            self.q_action.put(actions)
            self.q_action.task_done()
        except:
            print "__take_action: get action from queue failed."
            return

        if self.is_simulator_up and self.car_started:
            # print "__take_action: {}, q len {}".format(
            #     actions, self.q_action.qsize()
            # )
            map(
                lambda args: args[0].publish(args[1]),
                zip(self.action_pubs, actions)
            )
        else:
            print "__take_action: simulator up ({}), car started ({})".format(
                self.is_simulator_up, self.car_started)

    def __enque_done(self, data):
        if not self.q_ready.is_set():
            print "[__enque_exp]: queue not ready."
            return

        done = not data.data
        try:
            self.q_done.put(done, timeout=0.1)
        except:
            # print "__enque_done: q_done full."
            pass
        # print "__eqnue_done: {}".format(done)

    def __heartbeat_checker(self, data):
        print "Heartbeat signal: {}, First time: {}".format(
            data, self.first_time
        )
        if not data.data and not self.first_time:
            self.terminatable = True
        else:
             pass
        self.first_time = False


