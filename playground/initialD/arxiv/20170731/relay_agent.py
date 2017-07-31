# -*- coding: utf-8 -*-
"""A dummy agent which listens and repeats action.

File name: relay_agent.py
Author: Jingchu Liu
Last Modified: July 29, 2017
"""

# Basic python
import time
# Threading and Multiprocessing
from multiprocessing import JoinableQueue as Queue
# ROS
import rospy
from std_msgs.msg import Int16


class RelayAgent(object):
    """Dummy relay agent.

    This agent subscribe to the '/decision_result' topic from environment
    and put the received data into a queue. Then it retrieves and replay
    the queued data when the its `step()` method is callled.
    """
    def __init__(self, queue_len):
        self.q = Queue(queue_len)
        rospy.Subscriber('/decision_result', Int16, self.__mover)

    def __mover(self, data):
        try:
            key = data.data
            # print "[__mover]: {}".format(key)
            if self.q.full():
                self.q.get_nowait()
                self.q.task_done()
            self.q.put(key, timeout=0.1)
        except Exception as e:
            print "[__mover]: action enque failed. {}".format(e.message)
        return

    def step(self, *args, **kwargs):
        while True:
            try:
                action = self.q.get(timeout=0.1)
                self.q.task_done()
                break
            except:
                print "[step]: get action failed"
                time.sleep(0.1)
        # print "[step()]: action = {}".format(action)
        return action, {}

    def set_session(self, *args):
        return


