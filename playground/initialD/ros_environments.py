# -*- coding: utf-8 -*-
import multiprocessing
import logging
from Queue import Queue

from scipy.misc import imresize

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Float32
from sensor_msgs.msg import Image

from timer import Timer


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

        # self.node_process = multiprocessing.Process(target=self.__new_node)
        # self.node_process.start()

        self.__new_node()

        self.heart_beat_listener = rospy.Subscriber(
            '/rl/is_running', Bool,
            callback=self.__monitor_node
        )

    def __new_node(self):
        # ROS node
        rospy.init_node('DrivingSimulatorEnv')
        self.node = DrivingSimulatorNode(
            self.queue_observation, self.queue_reward, self.queue_action,
            self.defs_observation, self.defs_reward, self.defs_action,
            self.rate_action
        )

    def __monitor_node(self, data):
        is_running = data.data
        if not is_running:
            print "Shutdown node!"
            # self.node.shutdown()
            # self.node_process.terminate()
        else:
            print "Opening node!"
            pass
            # self.__new_node()
            # self.node_process = multiprocessing.Process(target=self.__new_node)
            # self.node_process.start()

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


class DrivingSimulatorNode(
    object
    # multiprocessing.Process
):
    def __init__(self, queue_observation, queue_reward, queue_action,
                 defs_observation, defs_reward, defs_action,
                 rate_action):
        #super(DrivingSimulatorNode, self).__init__()
        self.queue_observation = queue_observation
        self.queue_reward = queue_reward
        self.queue_action = queue_action

        self.defs_observation = defs_observation
        self.defs_reward = defs_reward
        self.defs_action = defs_action

    # def run(self):
        self.brg = CvBridge()

        # subscribers
        f_subs = lambda defs: message_filters.Subscriber(defs[0], defs[1])
        self.ob_subs = map(f_subs, defs_observation)
        self.reward_subs = map(f_subs, defs_reward)
        # sync obs and reward subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.ob_subs + self.reward_subs, 10, 0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.__enque_exp)

        # publishers
        f_pubs = lambda defs: rospy.Publisher(
            defs[0], defs[1], queue_size=10, latch=True
        )
        self.action_pubs = map(f_pubs, defs_action)
        self.actor_loop = Timer(
            rospy.Duration(1.0/rate_action), self.__take_action
        )

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
       actions = self.queue_action.get()
       self.queue_action.put(actions)
       self.queue_action.task_done()
       print "__take_action: {}, q len {}".format(actions,
                                                  self.queue_action.qsize())

       map(lambda args: args[0].publish(args[1]),
           zip(self.action_pubs, actions))


