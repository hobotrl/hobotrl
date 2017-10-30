"""Road validity monitor and publisher.

This script setup a ROS node to monitor realtime signal status and update the
validity of the corresponding road. The validity of the road that ego car is
currently on is published periodically.

Related ROS topics:
The signals status are derived from the topic `/signals`. The current road is
subscribed from '/rl/current_road'. And the current road validity is published
to'/rl/current_road_validity'.

The correspondence between signals and link roads are loaded from a pickled file
generated by `parse_map.py`.

:author: Jingchu LIU
:date: 2017-10-17
"""

from os.path import dirname, realpath
import sys
import argparse
import cPickle
import rospy
from std_msgs.msg import Bool, Int16, String
from autodrive_msgs.msg import Signals
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from ros_environments.utils.timer import Timer

class RoadValidity(object):
    def __init__(self, path_to_file):
        self.dict_road_juncdir_stopline, \
        self.dict_road_juncdir_signal, \
        self.dict_stopline_signal, \
        self.dict_link_signal, \
        self.dict_signal_links = cPickle.load(open(path_to_file, 'rb'))

        self.link_validity = {}
        for link in self.dict_link_signal:
            self.link_validity[link] = 0  # default is True

        self.current_road = None
        self.cnt = 0

        rospy.init_node('Road_Validity')

        rospy.Subscriber('/signals', Signals, self.update_validity)
        rospy.Subscriber('/rl/current_road', String, self.update_current_road)
        self.pub_validity = rospy.Publisher(
            '/rl/current_road_validity', Int16, queue_size=10)
        self.pub_road_change = rospy.Publisher(
            '/rl/current_road_change', Bool, queue_size=10, latch=True)
        self.pub_intersection = rospy.Publisher(
            '/rl/entering_intersection', Bool, queue_size=10, latch=True)
        Timer(rospy.Duration(1/20.0), self.pub_current_validity)

    def update_validity(self, data):
        for signal in data.signals:
            signal_id = signal.id
            color = signal.color
            for link_id in self.dict_signal_links[signal_id]:
                self.link_validity[link_id] = color

    def update_current_road(self, data):
        if data.data != self.current_road:
            self.pub_road_change.publish(True)
            self.pub_intersection.publish(data.data == 'L')
            self.current_road = data.data
            self.cnt = 10
        elif self.cnt > 0:
            self.pub_road_change.publish(True)
            self.pub_intersection.publish(self.current_road[0]=='L')
            self.cnt -= 1
        else:
            self.pub_road_change.publish(False)
            self.pub_intersection.publish(False)

    def pub_current_validity(self, data):
        if self.current_road is None or self.current_road[0] != 'L':
            self.pub_validity.publish(0)
        else:
            self.pub_validity.publish(self.link_validity[self.current_road])

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Road validity signal publisher node.')
    parser.add_argument('info_dir')
    args = parser.parse_args()
    path_to_file = args.info_dir

    print '[road_validity]: initializing node.'
    try:
        node = RoadValidity(path_to_file)
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        print '[road_validity]: node terminiated.'