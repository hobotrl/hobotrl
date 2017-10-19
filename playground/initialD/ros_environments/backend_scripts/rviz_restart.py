"""Launch file restarter and manager.
This script manages the ROS launch file. It starts and terminates a user
specified launch file on ROS topic 'simulator_restart' and publishes a
heartbeat signal indicating the state of simulator. The script also listens
to various topics to automatically shutdown and simulator.

:author: Gang XU, Jingchu LIU
:data: 2017-09-06
"""

import time
import signal
import argparse
from collections import deque
import subprocess
import numpy as np
from numpy import linalg as LA
import rospy
from std_msgs.msg import Char, Int16, Bool
from autodrive_msgs.msg import CarStatus
from base import BaseEpisodeMonitor


class SimulatorEpisodeMonitor(BaseEpisodeMonitor):
    def __init__(self, launch_name):
        """Initialization.

        :param launch_name: name of the launch file for planning.
        """
        # === Init super class ===
        super(SimulatorEpisodeMonitor, self).__init__()

        # === Subprocess related ===
        self.launch_name = launch_name
        self.process_list = list()
        self.process_names = [
            ['roslaunch', 'planning', self.launch_name]]
        print "[rviz_restart]: using launch file {}".format(self.launch_name)

        # === Simulator states ===
        self.last_pos = deque(maxlen=2000) # list of last 2000 position points. Approx. 40 secs @ 50Hz
        self.last_on_opposite_path = 1  # last latched signal value for `on_opposite_path`

        # periodic opposite path signal
        self.opposite_path_pub = rospy.Publisher(
            "/rl/last_on_opposite_path", Int16, queue_size=10, latch=True)

        # subscribers
        rospy.Subscriber('/error/type', Int16, self.__car_out_of_lane_callback)
        rospy.Subscriber('/car/status', CarStatus, self.__car_status_callback)
        rospy.Subscriber('/rl/on_grass', Int16, self.__car_out_of_lane_callback)
        rospy.Subscriber('/rl/on_opposite_path', Int16, self.__assign_last_op_callback)

    def _terminate(self):
        """shutdown launch file processes."""
        if len(self.process_list) is 0:
            print("[SimulatorEpisodeMonitor._terminate()]: no process to terminate")
        else:
            for p in self.process_list:
                p.send_signal(signal.SIGINT)
                while p.poll() is None:
                    print (
                        "[SimulatorEpisodeMonitor._terminate()]: "
                        "simulator process {} termination in progress..."
                    ).format(p.pid)
                    time.sleep(1.0)
                print (
                     "[SimulatorEpisodeMonitor._terminate()]: "
                    "simulator proc {} terminated with exit code {}"
                ).format(p.pid, p.returncode)
            self.process_list = []
            print("[SimulatorEpisodeMonitor]: termination done!")

        return

    def _start(self):
        """Restart nodes specified in a list of commands."""
        for name in self.process_names:
            p = subprocess.Popen(name)
            self.process_list.append(p)
        print("[rviz_restart.restart]: started launch file!")

        return

    def __car_out_of_lane_callback(self, data):
        """Various reasons car is out of lane (e.g. on grass)."""
        if abs(int(data.data)-1)<0.001:
            rospy.logwarn("Car out of Lane! (on grass)")
            self.terminate()

    def __car_status_callback(self, data):
        # TODO:
        #  The car stops or go with a very slow speed, doesn't mean the car
        #  arrived at the destination. maybe we can set a circle near
        #  destination, when it reaches that range, we can restart the process 
        if self.is_running:
            # detect destination
            try:
                dest_x = rospy.get_param('/car/dest_coord_x')
                dest_y = rospy.get_param('/car/dest_coord_y')
                dest_dist = np.sqrt(np.square(data.position.x-dest_x) +
                                    np.square(data.position.y-dest_y))
                if dest_dist < 30:
                    rospy.logwarn("Ego car reached destination.")
                    self.last_pos.clear()
                    self.terminate()
            except:
                pass
            # detect stopping for too long
            self.last_pos.append(np.array([data.position.x, data.position.y, data.position.z]))
            if (len(self.last_pos)==self.last_pos.maxlen and
                LA.norm(self.last_pos[0] - self.last_pos[-1])<0.1):
                rospy.logwarn("The car stops moving!")
                self.last_pos.clear()
                self.terminate()
        else:
            pass

    def __assign_last_op_callback(self, data):
        self.last_on_opposite_path = data.data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rviz_restart.py')
    parser.add_argument('launch_name', type=str)
    args = parser.parse_args()
    try:
        mon = SimulatorEpisodeMonitor(args.launch_name)
        mon.spin()
    except rospy.ROSInterruptException:
        pass
