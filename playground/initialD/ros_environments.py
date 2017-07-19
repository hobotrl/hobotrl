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
    def __init__(self,
                 defs_observation, defs_reward, defs_action,
                 rate_action, buffer_sizes):
        """Initialization.
        :param topics_observation:
        :param topics_reward:
        :param topics_action:
        :param rate_action:
        :param buffer_sizes:
        """
        # CDC queues for obs, reward, and action
        self.queue_observation = Queue(buffer_sizes['observation'])
        self.queue_reward = Queue(buffer_sizes['reward'])
        self.queue_action = Queue(buffer_sizes['action'])
        self.defs_observation = defs_observation
        self.defs_reward = defs_reward
        self.defs_action = defs_action
        self.rate_action = rate_action
        self.parent_conn, self.child_conn = Pipe()

        self.daemon = multiprocessing.Process(target=self.node_daemon,
                                              args=(self.parent_conn,))
        self.daemon.start()

    def shutdown(self):
        print "Shutting down env."
        self.daemon.terminate()
        self.daemon.join()

    def node_daemon(self, conn):
        """Reverse poison pill."""
        while True:
            print "Daemon starting new node"
            self.__new_node()
            try:
                print "Daemon waiting for poison pill"
                if_terminate = conn.recv()
                print "Daemon received signal {}".format(if_terminate)
                if if_terminate:
                    print "Terminating node process: {}. PID: {}".format(
                        self.node.name, self.node.pid),
                    self.node.terminate()
                    self.node.join()
                    print "Done!"
                    continue
                else:
                    raise ValueError('Unrecognized signal!')
            except EOFError:
                raise ValueError('Other end terminated!')

    def __new_node(self):
        self.node = DrivingSimulatorNode(
            self.queue_observation, self.queue_reward, self.queue_action,
            self.defs_observation, self.defs_reward, self.defs_action,
            self.rate_action, self.child_conn
        )
        self.node.daemon = True
        self.node.start()

    def step(self, action):
        # put action to action queue
        if self.queue_action.full():
            self.queue_action.get()
            self.queue_action.task_done()
        self.queue_action.put(action)
        print "step: action: {}, queue size: {}".format(
            action, self.queue_action.qsize()
        )
        # decide if episode is done
        done = False  # TODO: realize a mechanism to determinine episode 
        # compile observation and reward from queued experiences
        next_state = self.queue_observation.get()[0]
        reward = self.queue_reward.get()
        print "step: {}".format(reward)
        reward = sum(map(float, reward))
        self.queue_observation.task_done()
        self.queue_reward.task_done()
        # other stuff
        info = None

        return next_state, reward, done, info

    def reset(self):
        state = self.queue_observation.get()[0]
        self.queue_observation.task_done()
        return state


class DrivingSimulatorNode(multiprocessing.Process):
    def __init__(self,
                 queue_observation, queue_reward, queue_action,
                 defs_observation, defs_reward, defs_action,
                 rate_action, conn):
        super(DrivingSimulatorNode, self).__init__()

        self.queue_observation = queue_observation
        self.queue_reward = queue_reward
        self.queue_action = queue_action

        self.defs_observation = defs_observation
        self.defs_reward = defs_reward
        self.defs_action = defs_action

        self.rate_action = rate_action

        self.conn = conn

        self.first_time = True 

    def run(self):
        print "Started process: {}".format(self.name)
        rospy.init_node('Node_DrivingSimulatorEnv_Simulator')

        self.heart_beat_listener = rospy.Subscriber(
            '/rl/is_running', Bool,
            callback=self.__daemon_conn
        )

        self.restart_pub = rospy.Publisher(
            '/rl/simulator_restart', Bool, queue_size=10, latch=True
        )

        self.brg = CvBridge()

        # subscribers
        f_subs = lambda defs: message_filters.Subscriber(defs[0], defs[1])
        self.ob_subs = map(f_subs, self.defs_observation)
        self.reward_subs = map(f_subs, self.defs_reward)
        # sync obs and reward subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.ob_subs + self.reward_subs, 10, 0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.__enque_exp)

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
        rospy.spin()
        print "Returning from run in process: {}".format(self.name)

    def __enque_exp(self, *args):
        if self.queue_observation.full():
            self.queue_observation.get()
            self.queue_observation.task_done()
        if self.queue_reward.full():
            self.queue_reward.get()
            self.queue_reward.task_done()
        num_obs = len(self.ob_subs)
        num_reward = len(self.reward_subs)
        # TODO: test
        args = list(args)
        args[0] = imresize(
            self.brg.imgmsg_to_cv2(args[0], 'rgb8'),
            (640, 640)
        )
        self.queue_observation.put(
            (args[:num_obs])
        )
        self.queue_reward.put(
            (map(lambda data: data.data, args[num_obs:]))
        )
        print "__enque_exp: {}".format(args[num_obs:])

    def __take_action(self, data):
       # actions = self.queue_action[-1]
       actions = self.queue_action.get()
       self.queue_action.put(actions)
       self.queue_action.task_done()
       print "__take_action: {}, q len {}".format(
           actions, self.queue_action.qsize()
       )

       map(
           lambda args: args[0].publish(args[1]),
           zip(self.action_pubs, actions)
       )

    def __daemon_conn(self, data):
        print "Heartbeat signal: {}, First time: {}".format(
            data, self.first_time
        )
        if not data.data and not self.first_time:
            # rospy.signal_shutdown("Heart beat = False")
            self.conn.send(True)
        else:
             pass
        self.first_time = False


