"""Launch file restarter and manager.
This script manages the ROS launch file. It starts and terminates a user
specified launch file on ROS topic 'simulator_restart' and publishes a
heartbeat signal indicating the state of simulator. The script also listens
to various topics to automatically shutdown and simulator.

:author: Gang XU, Jingchu LIU
:data: 2017-09-06
"""

import sys
import os
from os.path import dirname, realpath
import time
import signal
import argparse
from collections import deque
import subprocess
import numpy as np
from numpy import linalg as LA
import rospy
import rospkg
from std_msgs.msg import Char, Int16, Bool
from autodrive_msgs.msg import CarStatus
# append initialD path to PATH
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from ros_environments.utils.timer import Timer

class restart_ros_launch:
    def __init__(self, launch_name):
        """Initialization.

        :param launch_name: name of the launch file for planning.
        """
        # Subprocess related
        self.launch_name = launch_name
        self.process_list = list()
        self.process_names = [
            ['roslaunch', 'planning', self.launch_name]]
        print "[rviz_restart]: using launch file {}".format(self.launch_name)

        # Simulator states
        self.is_running = False  # is simulator curretnly running?
        self.last_pos = deque(maxlen=1000) # list of last 1000 position points. Approximately 20 secs @ 50Hz
        self.last_on_opposite_path = 1  # last latched signal value for `on_opposite_path`

        # ROS node
        rospy.init_node('LaunchFileRestarter')
        # publishers
        # async signal for simulator state
        self.is_running_pub = rospy.Publisher(
            "/rl/is_running", Bool, queue_size=10, latch=True)
        # periodic heartbeat
        self.heartbeat_pub = rospy.Publisher(
            "/rl/simulator_heartbeat", Bool, queue_size=10, latch=True)
        # opposite path
        self.opposite_path_pub = rospy.Publisher(
            "/rl/last_on_opposite_path", Int16, queue_size=10, latch=True)
        # periodic heartbeat
        Timer(rospy.Duration(1/20.0),
              lambda *args: self.heartbeat_pub.publish(self.is_running))
        # periodic opposite path signal
        Timer(rospy.Duration(1/20.0),
              lambda *args:
              self.opposite_path_pub.publish(self.last_on_opposite_path))

        # subscribers
        rospy.Subscriber('/rl/simulator_restart', Bool, self.__restart_callback)
        rospy.Subscriber('/error/type', Int16, self.__car_out_of_lane_callback)
        rospy.Subscriber('/car/status', CarStatus, self.__car_not_move_callback)
        rospy.Subscriber('/rl/on_grass', Int16, self.__car_out_of_lane_callback)
        rospy.Subscriber('/rl/on_opposite_path', Int16, self.__assign_last_op_callback)

    def spin(self):
        rospy.spin()
        self.terminate()

    def terminate(self):
        # flush heartbeat = False for 1 sec
        self.is_running = False
        secs = 1
        while secs != 0:
            print "[rviz_restart.terminate]: Shutdown simulator nodes in {} secs".format(secs)
            secs -= 1
            time.sleep(1.0)

        # shutdown simulator node
        if len(self.process_list) is 0:
            print("[rviz_restart.terminate]: no process to terminate")
        else:
            rospy.loginfo("now shut down launch file")
            for p in self.process_list:
                # p.terminate()
                # p.kill()
                p.send_signal(signal.SIGINT)
                while p.poll() is None:
                    print (
                        "[rviz_restart.terminate]: Simulator proc {} termination in progress..."
                    ).format(p.pid)
                    time.sleep(1.0)
                print (
                    "[rviz_restart.terminate]: Simulator proc {} terminated with exit code {}"
                ).format(p.pid, p.returncode)
            self.process_list = []
            print("Done!")

        # signal env node shutdown
        print "[rviz_restart.terminate]: publish heartbeat=False!"
        self.is_running_pub.publish(False)

    def __restart_callback(self, data):
        """Terminate and restart simulator on command."""
        print "[rviz_restart.restart]: restart callback with {}".format(data.data)
        # terminates simulator on False
        if data.data==False:
            print "[rviz_restart.restart]: mere termination requested."
            self.terminate()
            print "[rviz_restart.restart]: termination finished."
            self.is_running_pub.publish(False)
            return
        # restart on True
        else:
            # restart launch file
            for name in self.process_names:
                p = subprocess.Popen(name)
                self.process_list.append(p)
            print("[rviz_restart.restart]: restarted launch file!")
            self.is_running = True
            self.is_running_pub.publish(True)
            print "[rviz_restart.restart]: publish heartbeat=True!"

    def __car_out_of_lane_callback(self, data):
        """Various reasons car is out of lane. (On grass for one)"""
        if abs(int(data.data)-1)<0.001:
            rospy.logwarn("Car out of Lane! (on grass)")
            self.terminate()

    def __car_not_move_callback(self, data):
        # TODO:
        #  The car stops or go with a very slow speed, doesn't mean the car
        #  arrived at the destination. maybe we can set a circle near
        #  destination, when it reaches that range, we can restart the process 
        self.last_pos.append(np.array([data.position.x, data.position.y, data.position.z]))
        if (len(self.last_pos)==self.last_pos.maxlen and
            LA.norm(self.last_pos[0] - self.last_pos[-1])<0.1):
            rospy.logwarn("The car stops moving!")
            self.last_pos.clear()
            self.terminate()

    def __assign_last_op_callback(self, data):
        self.last_on_opposite_path = data.data



if __name__ == '__main__':
    parser = argparse.ArgumentParser('rviz_restart.py')
    parser.add_argument('launch_name', type=str)
    args = parser.parse_args()
    try:
        myobjectx = restart_ros_launch(args.launch_name)
        myobjectx.spin()
    except rospy.ROSInterruptException:
        pass
