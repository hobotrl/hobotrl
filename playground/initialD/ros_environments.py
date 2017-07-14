# -*- coding: utf-8 -*-
import os
import signal
from time import time, sleep
from Queue import Queue
import multiprocessing

import rospy
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
        self.queue_observation = Queue(buffer_sizes['observation'])
        self.queue_reward = Queue(buffer_sizes['reward'])
        self.queue_action = Queue(buffer_sizes['action'])

        rospy.init_node('DrivingSimulatorEnv')

        f_subs = lambda defs: message_filters.Subscriber(defs[0], defs[1])
        self.ob_subs = map(f_subs, defs_observation)
        self.reward_subs = map(f_subs, defs_reward)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.ob_subs + self.reward_subs, 10, 0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.__enque_exp)

        f_pubs = lambda defs: rospy.Publisher(
            defs[0], defs[1], queue_size=buffer_sizes['action'], latch=True
        )
        self.action_pubs = map(f_pubs, defs_action)
        self.action_rate = rospy.Rate(rate_action)
        self.actor_loop = rospy.Timer(
            rospy.Duration(1.0/rate_action), self.__take_action
        )
        print "Env initialized."

    def step(self, action):
        # put action to action queue
        if self.queue_action.full():
            self.queue_action.get()
            self.queue_action.task_done()
        self.queue_action.put(action)
        # decide if episode is done
        done = False  # TODO: realize a mechanism to determinine episode 
        # compile observation and reward from queued experiences
        print "Getting next_state and reward..."
        next_state = self.queue_observation.get()
        reward = self.queue_reward.get()
        self.queue_observation.task_done()
        self.queue_reward.task_done()
        # other stuff
        info = None

        return next_state, reward, done, info

    def __enque_exp(self, *args):
        if self.queue_observation.full():
            self.queue_observation.get()
            self.queue_observation.task_done()
        if self.queue_reward.full():
            self.queue_reward.get()
            self.queue_reward.task_done()
        num_obs = len(self.ob_subs)
        num_reward = len(self.reward_subs)
        self.queue_observation.put((args[:num_obs]))
        self.queue_reward.put((args[num_obs:]))

    def __take_action(self, data):
        #while not rospy.is_shutdown():
        if True:
           print 'Getting action',
           actions = self.queue_action.get()
           self.queue_action.put(actions)
           self.queue_action.task_done()
           print 'Done'
           # actions = [1.0]
           # print self.action_pubs[0].name
           self.action_pubs[0].publish(actions[0])
           # map(lambda ac, pub: pub.publish(ac),
           #      zip(actions, self.action_pubs))
           # self.action_rate.sleep()
           sleep(0.1)


if __name__ == "__main__":
    env = DrivingSimulatorEnv(
        [('/training/image', Image)],
        [('/rl/has_obstacle_nearby', Bool),
         ('/rl/distance_to_longestpath', Float32)],
        [('/rl/action/test', Float32)],
        rate_action=1,
        buffer_sizes={'observation': 10, 'reward': 10, 'action': 10}
    )
    try:
        print env
        env.step((1.0, ))
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
