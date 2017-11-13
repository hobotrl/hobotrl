"""Monitors the status of nodes and publish signal if they are died.
:author: Jingchu LIU
:date: 2017-11-10
"""
import argparse
import logging
import rospy
from rospy.timer import Timer
import rosnode
from std_msgs.msg import Bool

class NodeMonitor(object):
    """ROS node status monitor.
    Monitors
    """
    def __init__(self, node_names, rate, topic_prefix='/rl/node_up'):
        self._node_names = node_names
        self._rate = rate
        self._state = {name: None for name in self._node_names}
        rospy.init_node('node_monitor')
        self.pubs = {
            name: rospy.Publisher(
                topic_prefix + name, Bool, queue_size=10, latch=True
            ) for name in self._node_names
        }
        self.monitor_loop = Timer(
            rospy.Duration(nsecs=int(1.0/self._rate*1e9)),
            self.monitor_once
        )

    def monitor_once(self, dummy):
        up_names = rosnode.get_node_names()
        for name in self._node_names:
            if name in up_names:
                if self._state[name] is None:
                    self._state[name] = True
            else:
                if self._state[name]:
                    self._state[name] = False
            self.pubs[name].publish(
                self._state[name] is None or self._state[name]
            )

    @staticmethod
    def spin():
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('node_monitor.py')
    parser.add_argument('--node_name', nargs='*')
    parser.add_argument('--rate', nargs='?', type=float, default='1.0')
    args = parser.parse_args()
    logging.warning(
        "[node_monitor]: "
        "start to monitor nodes: {} with rate {}".format(
            args.node_name, args.rate
        )
    )
    try:
        mon = NodeMonitor(node_names=args.node_name, rate=args.rate)
        mon.spin()
    except rospy.ROSInterruptException:
        pass
    logging.warning(
        "[node_monitor]: out."
    )