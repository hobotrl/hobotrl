# system
import signal
import time
import sys
import argparse
import subprocess
# data structure
from collections import deque
import numpy as np
# ROS
import rospy
from rospy.timer import Timer
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Char
from nav_msgs.msg import Path
from autodrive_msgs.msg import Control, CarStatus, PlanningTraj
# HobotRL
from base import BaseEpisodeMonitor
from utils.toy_car_reset import get_trajectory, traj_plan_to_nav


class ToyCarEpisodeMonitor(BaseEpisodeMonitor):
    def __init__(self, launch_name):
        """Initialization.

        :param launch_name: name of the launch file.
        """
        # === Init super class ===
        super(ToyCarEpisodeMonitor, self).__init__()

        # === Subprocess related ===
        self.launch_name = launch_name
        self.process_list = list()
        self.process_names = [
            ['roslaunch', 'planning', self.launch_name]]
        print "[toycar_restart]: using launch file {}".format(self.launch_name)

        # === Simulator states ===
        self.last_pos = deque(maxlen=500) # list of last 500 position points. Approx. 10 secs @ 50Hz

        self.control_status = False
        self.car_status = None

        # additional publishers and subscribers
        self.pub_traj_plan = rospy.Publisher(
            "/planning/trajectory", PlanningTraj, queue_size=1, latch=True)
        self.pub_traj_viz = rospy.Publisher(
            "/visualization/trajectory", Path, queue_size=1, latch=True)
        self.pub_keyboardmode = rospy.Publisher(
            '/autoDrive_KeyboardMode', Char, queue_size=1
        )
        self.sub_car_status = rospy.Subscriber(
            '/car/status', CarStatus, self._log_car_status
        )
        self.sub_car_control = rospy.Subscriber(
            '/car/control', Control, self._set_control_status)
        self.perimeter_checker = Timer(
            rospy.Duration(0.1), self._check_perimeter)

    def _terminate(self):
        """shutdown launch file processes."""
        if len(self.process_list) is 0:
            print("[ToyCarEpisodeMonitor._terminate()]: no process to terminate")
        else:
            for p in self.process_list:
                p.send_signal(signal.SIGINT)
                while p.poll() is None:
                    print (
                        "[ToyCarEpisodeMonitor._terminate()]: "
                        "backend process {} termination in progress..."
                    ).format(p.pid)
                    time.sleep(1.0)
                print (
                     "[ToyCarEpisodeMonitor._terminate()]: "
                    "backend process {} terminated with exit code {}"
                ).format(p.pid, p.returncode)
            self.process_list = []
            print("[ToyCarEpisodeMonitor]: termination done!")

        return

    def _start(self):
        """Restart nodes specified in a list of commands."""
        for name in self.process_names:
            p = subprocess.Popen(name)
            self.process_list.append(p)
        print("[toycar_restart._start]: started launch file!")

        print("[toycar_restart._start]: resetting toy car.")
        self._mount_control()
        self.reset_to(
            (128, 128, 1.0), max_speed=20, turning_radius=15, step_size=0.1,
            max_length=100
        )
        self._unmount_control()
        return

    def _mount_control(self):
        while not self.control_on:
            self.pub_keyboardmode.publish(ord(' '))
            time.sleep(1.0)

    def _unmount_control(self):
        while self.control_on:
            self.pub_keyboardmode.publish(ord(' '))
            time.sleep(1.0)

    def _set_control_status(self, data):
        self.control_on = data.autodrive_mode

    def _log_car_status(self, data):
        if data is not None:
            self.car_status = data

    def _check_perimeter(self, data):
        x, y = self.car_status.position.x, self.car_status.position.y
        if x < 20 or x > 240 or y < 20 or y > 240:
            print "[ToyCarEpisodeMonitor]: crossed fence at ({}, {})".format(
                x, y
            )
            self.terminate()

    def reset_to(self, goal, max_speed=1.0,
                 turning_radius=10.0, step_size=0.1, max_length=50):
        if self.car_status is None:
            print "Not getting any car_status."
            return False
        else:
            current_pose = (
                self.car_status.position.x,
                self.car_status.position.y,
                euler_from_quaternion((
                    self.car_status.orientation.x,
                    self.car_status.orientation.y,
                    self.car_status.orientation.z,
                    self.car_status.orientation.w,
               ))[2]
            )
            current_speed = self.car_status.speed

            cnt_plan = 100
            while True:
                self.plan_once(current_pose, goal, current_speed, max_speed,
                               turning_radius, step_size, max_length)
                time.sleep(1.0)
                current_pose = (
                    self.car_status.position.x,
                    self.car_status.position.y,
                    euler_from_quaternion((
                        self.car_status.orientation.x,
                        self.car_status.orientation.y,
                        self.car_status.orientation.z,
                        self.car_status.orientation.w,
                    ))[2]
                )
                current_speed = self.car_status.speed
                cnt_plan -= 1
                diff = np.array(goal) - np.array(current_pose)
                err1 = np.sqrt(np.sum(np.square(diff[:2])))
                err2 = np.abs(
                    (diff[2] + np.pi) % (2 * np.pi) - np.pi
                )  # warped to [-pi, +pi)
                print "N plans left: {}".format(cnt_plan)
                print "Errors: pos {} angle {}".format(err1, err2)

                if err1 < 30 and err2 < 0.3:
                    print "Reset finished"
                    break
                elif cnt_plan <= 0:
                    print "Reset timeout!"
                    break
                else:
                    print goal,
                    print current_pose,
                    print current_speed

    def plan_once(self, start, goal, start_speed, max_speed,
                  turning_radius=10.0, step_size=0.1, max_length=50):
        traj_plan = get_trajectory(start, goal, start_speed, max_speed,
                                   turning_radius, step_size, max_length)
        traj_viz = traj_plan_to_nav(traj_plan)
        self.pub_traj_plan.publish(traj_plan)
        self.pub_traj_viz.publish(traj_viz)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('toycar_restart.py')
    parser.add_argument('launch_name', type=str)
    args = parser.parse_args()
    try:
        mon = ToyCarEpisodeMonitor(args.launch_name)
        mon.spin()
    except rospy.ROSInterruptException:
        pass