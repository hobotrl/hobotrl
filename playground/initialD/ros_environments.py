# -*- coding: utf-8 -*-
# Basic python
import time
# Multi-process
import multiprocessing
from multiprocessing import JoinableQueue as Queue
from multiprocessing import Pipe
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
        self.q_obs = Queue(buffer_sizes['obs'])
        self.q_reward = Queue(buffer_sizes['reward'])
        self.q_action = Queue(buffer_sizes['action'])
        self.q_done = Queue(1)
        # pub and sub definitions
        self.defs_obs = defs_obs
        self.defs_reward = defs_reward
        self.defs_action = defs_action
        self.rate_action = rate_action
        # daemon processes
        self.proc_monitor = multiprocessing.Process(target=self.monitor)
        self.proc_monitor.start()

    def monitor(self):
        while True:
            print "Monitor: running new node."
            self.__run_node()
            print "Monitor: finished running node."

    def __run_node(self):
        node = DrivingSimulatorNode(
            self.q_obs, self.q_reward, self.q_action, self.q_done,
            self.defs_obs, self.defs_reward, self.defs_action,
            self.rate_action
        )
        node.start()
        node.join()

    def step(self, action):
        # enqueue action
        if self.q_action.full():
            self.q_action.get()
            self.q_action.task_done()
        self.q_action.put(action)
        print "step: action: {}, queue size: {}".format(
            action, self.q_action.qsize()
        )
        # compile observation
        next_state = self.q_obs.get()[0]
        self.q_obs.task_done()
        # calculate reward
        rewards = self.q_reward.get()
        self.q_reward.task_done()
        print "step(): rewards {}".format(rewards)
        reward = -100.0 * float(rewards[0]) + \
                 -10.0 * float(rewards[1]) + \
                 1.0 * float(rewards[2]) + \
                 -100.0 * (1 - float(rewards[3]))
        # decide if episode is done
        try:
            done = self.q_done.get(False)
            self.q_done.task_done()
        except:
            print "step(): queue_done empty."
            done = True  # assume episode done if queue_done is emptied
        # info
        info = None

        print "step(): reward {}, done {}".format(reward, done)
        return next_state, reward, done, info

    def reset(self):
        state = self.q_obs.get()[0]
        self.q_obs.task_done()
        return state


class DrivingSimulatorNode(multiprocessing.Process):
    def __init__(self, q_obs, q_reward, q_action, q_done,
                 defs_obs, defs_reward, defs_action,
                 rate_action):
        super(DrivingSimulatorNode, self).__init__()

        self.q_obs = q_obs
        self.q_reward = q_reward
        self.q_action = q_action
        self.q_done = q_done

        self.defs_obs = defs_obs
        self.defs_reward = defs_reward
        self.defs_action = defs_action

        self.rate_action = rate_action

        self.first_time = True
        self.terminatable = False

    def run(self):
        print "Started process: {}".format(self.name)
        rospy.init_node('Node_DrivingSimulatorEnv_Simulator')

        self.heartbeat_listener = rospy.Subscriber(
            '/rl/is_running', Bool,
            callback=self.__daemon_conn
        )

        self.restart_pub = rospy.Publisher(
            '/rl/simulator_restart', Bool, queue_size=10, latch=True
        )

        self.brg = CvBridge()

        # subscribers
        f_subs = lambda defs: message_filters.Subscriber(defs[0], defs[1])
        self.ob_subs = map(f_subs, self.defs_obs)
        self.reward_subs = map(f_subs, self.defs_reward)
        # sync obs and reward subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.ob_subs + self.reward_subs, 10, 0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.__enque_exp)
        rospy.Subscriber('/rl/simulator_heartbeat', Bool, self.__enque_done)

        # publishers
        f_pubs = lambda defs: rospy.Publisher(
            defs[0], defs[1], queue_size=10, latch=True
        )
        self.action_pubs = map(f_pubs, self.defs_action)
        self.actor_loop = Timer(
            rospy.Duration(1.0/self.rate_action), self.__take_action
        )
        time.sleep(1.0)
        print "Signaling simulator restart!"
        self.restart_pub.publish(True)

        print "Let's roll!"
        time.sleep(6.0)
        self.action_pubs[0].publish(ord('1'))
        time.sleep(0.5)
        self.action_pubs[0].publish(ord(' '))
        time.sleep(0.5)
        self.action_pubs[0].publish(ord('g'))

        while not self.terminatable:
            time.sleep(1.0)

        self.q_obs.close()
        self.q_reward.close()
        self.q_action.close()
        self.q_done.close()

        rospy.signal_shutdown('Simulator down')
        print "Returning from run in process: {}, PID: {}".format(
            self.name, self.pid)

    def __enque_exp(self, *args):
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
        print "__enque_exp: {}".format(args[num_obs:])

    def __take_action(self, data):
       actions = self.q_action.get()
       self.q_action.put(actions)
       self.q_action.task_done()
       print "__take_action: {}, q len {}".format(
           actions, self.q_action.qsize()
       )
       map(
           lambda args: args[0].publish(args[1]),
           zip(self.action_pubs, actions)
       )

    def __enque_done(self, data):
        done = not data.data
        if self.q_done.full():
            self.q_done.get()
            self.q_done.task_done()
        self.q_done.put(done)
        # print "__eqnue_done: {}".format(done)

    def __daemon_conn(self, data):
        print "Heartbeat signal: {}, First time: {}".format(
            data, self.first_time
        )
        if not data.data and not self.first_time:
            self.terminatable = True
        else:
             pass
        self.first_time = False


