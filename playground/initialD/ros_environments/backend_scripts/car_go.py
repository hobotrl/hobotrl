#!/usr/bin/env python
"""Let the car continue to go in case of stop.
:author: Jingchu LIU
:date: 2017-10-09
"""
import time
from os.path import dirname, realpath
import sys
import argparse

# ROS
import rospy
from rospy.timer import Timer
from std_msgs.msg import Char
from autodrive_msgs.msg import CarStatus
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

class CarGo:
    def __init__(self, is_dummy_action=False):
        """Initialization."""
        self.is_dummy_action = is_dummy_action
        self.ema_speed = 10.0
        # ROS related
        rospy.init_node('car_go')
        rospy.Subscriber('/car/status', CarStatus, self.car_status_callback)
        self.start_pub = rospy.Publisher(
            '/autoDrive_KeyboardMode', Char, queue_size=10)
        self.car_go_loop = Timer(rospy.Duration(5.0), self.car_go_callback)

    def car_status_callback(self, data):
        """Calculate the exponential moving average of car speed.

        :param data:
        :return:
        """
        if data is not None:
            try:
                self.ema_speed = 0.9*self.ema_speed + data.speed
            except:
                pass

    def car_go_callback(self, data):
        if self.ema_speed < 0.1:
            time.sleep(0.5)
            if self.is_dummy_action:
                print "[CarGo]: sending key '0' ..."
                self.start_pub.publish(ord('0'))
            else:
                print "[CarGo]: sending key '1' ..."
                self.start_pub.publish(ord('1'))
            time.sleep(0.5)
            print "[CarGo]: sending key 'g' ..."
            self.start_pub.publish(ord('g'))
            self.ema_speed = 10.0

        return

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('car_go.py')
    parser.add_argument(
        '--use-dummy-action', dest='is_dummy_action', action='store_true')
    parser.set_defaults(is_dummy_action=False)
    args = parser.parse_args()
    print "[car_go]: inside file with dummy_action {}.".format(args.is_dummy_action)
    try:
        cargo = CarGo(is_dummy_action=args.is_dummy_action)
        cargo.spin()
    except rospy.ROSInterruptException:
        pass
    print "[car_go]: out."
