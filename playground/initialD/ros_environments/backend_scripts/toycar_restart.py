# system
import signal
import time
import sys
import argparse
import subprocess
from multiprocessing import Event
# data structure
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pylab
# ROS
import rospy
from rospy.timer import Timer
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Char, Int8
from nav_msgs.msg import Path
from autodrive_msgs.msg import Control, CarStatus, PlanningTraj
# HobotRL
from base import BaseEpisodeMonitor
from utils.toy_car_reset import get_trajectory, traj_plan_to_nav

from autodrive_msgs.msg import Control


class ToyCarEpisodeMonitor(BaseEpisodeMonitor):
    def __init__(self):
        """Initialization.
        """
        # === Init super class ===
        super(ToyCarEpisodeMonitor, self).__init__()

        # === Subprocess related ===
        self.process_list = list()
        self.process_names = [
            ['roslaunch', 'control', 'toycar_control.launch'],]
        self.lock_autodrive_mode = Event()
        self.lock_autodrive_mode.clear()
        self.lock_perimeter = Event()
        self.lock_perimeter.clear()

        # === Simulator states ===
        self.last_autodrive_mode = None
        self.car_status = None
        self.car_x_ema = None
        self.car_y_ema = None
        self.car_hdg_ema = None
        self.planning_status = None

        # additional publishers and subscribers
        self.pub_traj_plan = rospy.Publisher(
            "/planning/trajectory", PlanningTraj, queue_size=1, latch=True)
        self.pub_traj_viz = rospy.Publisher(
            "/visualization/trajectory", Path, queue_size=1, latch=True)
        self.pub_keyboardmode = rospy.Publisher(
            '/autoDrive_KeyboardMode', Char, queue_size=1, latch=True)
        self.pub_test_reward = rospy.Publisher(
            '/rl/toycar_test_reward', Int8, queue_size=1)
        self.sub_car_status = rospy.Subscriber(
            '/car/status', CarStatus, self._log_car_status)
        self.sub_car_control = rospy.Subscriber(
            '/car/control', Control, self._log_autodrive_mode)
        self.sub_planning_traj = rospy.Subscriber(
            "/planning/trajectory", PlanningTraj, self._log_planning_status
        )
        self.perimeter_checker = Timer(
            rospy.Duration(0.1), self._check_perimeter)
        self.test_reward = Timer(
            rospy.Duration(0.1), lambda *args: self.pub_test_reward.publish(0))
        self.control_mounter = Timer(
            rospy.Duration(1.0), self._mount_control
        )

    def _terminate(self):
        return

    def _start(self):
        """Restart nodes specified in a list of commands."""
        self.lock_perimeter.set()
        for name in self.process_names:
            p = subprocess.Popen(name)
            self.process_list.append(p)
        print("[rviz_restart.restart]: started launch file!")
        print("[toycar_restart._start]: resetting toy car.")

        while True:
            ret = self.reset_to(
                (128, 128, 1.0), max_speed=25, turning_radius=20, step_size=0.1, max_length=500)
            if ret:
                break
            time.sleep(2.0)

            self._kill_launchfile()

        print("[toycar_restart._start]: toy car reset!.")

        self.lock_perimeter.clear()

        return

    def _mount_control(self, data=None):
        if self.last_autodrive_mode is not None:
            if not self.last_autodrive_mode and not self.lock_autodrive_mode.is_set():
                print "Control status is {}, setting True".format(self.last_autodrive_mode)
                self.lock_autodrive_mode.set()
                self.pub_keyboardmode.publish(ord(' '))
                time.sleep(2.0)
                self.lock_autodrive_mode.clear()
        else:
            print "Control status is None."
            time.sleep(5.0)

    def _log_autodrive_mode(self, data):
        if data is not None:
            self.last_autodrive_mode = data.autodrive_mode

    def _log_car_status(self, data):
        if data is not None:
            self.car_status = data
            hdg = euler_from_quaternion((
                    self.car_status.orientation.x,
                    self.car_status.orientation.y,
                    self.car_status.orientation.z,
                    self.car_status.orientation.w,))[2]
            hdg = (hdg + np.pi) % (2 * np.pi) - np.pi
            if self.car_hdg_ema is not None:
                self.car_hdg_ema = 0.5 * self.car_hdg_ema + 0.5 * hdg
            else:
                self.car_hdg_ema = hdg
            if self.car_x_ema is not None:
                self.car_x_ema = 0.5 * self.car_x_ema + 0.5 * self.car_status.position.x
            else:
                self.car_x_ema = self.car_status.position.x
            if self.car_y_ema is not None:
                self.car_y_ema = 0.5 * self.car_y_ema + 0.5 * self.car_status.position.y
            else:
                self.car_y_ema = self.car_status.position.y

    def _log_planning_status(self, data):
        self.planning_status = True

    def _check_perimeter(self, data):
        if not self.lock_perimeter.is_set() and self.car_status is not None:
            x, y = self.car_status.position.x, self.car_status.position.y
            if x < 50 or x > 200 or y < 50 or y > 200:
                print "[ToyCarEpisodeMonitor]: crossed fence at ({}, {})".format(
                    x, y
                )
                self.terminate()

    def reset_to(self, goal, max_speed=2.0,
                 turning_radius=10.0, step_size=0.1, max_length=50, retry=100):
        if self.car_status is None:
            print "Not getting any car_status."
            return False
        else:
            current_pose = (
                self.car_x_ema,
                self.car_y_ema,
                self.car_hdg_ema)
            current_speed = self.car_status.speed

            cnt_plan = retry
            print "See figure?!!!!=============="
            fig = plt.figure()
            plt.axis()
            plt.xlim((0, 255))
            plt.ylim((0, 255))
            plt.ion()
            while True:
                self.plan_once(current_pose, goal, current_speed, max_speed,
                               turning_radius, step_size, max_length)
                time.sleep(0.5)
                current_pose = (
                    self.car_x_ema,
                    self.car_y_ema,
                    self.car_hdg_ema
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

                if err1 < 50 and err2 < 5.0:
                    print "Reset finished"
                    break
                elif cnt_plan <= 0:
                    print "Reset timeout!"
                    break
                else:
                    print goal,
                    print current_pose,
                    print current_speed
            pylab.close()

            return True

    def plan_once(self, start, goal, start_speed, max_speed,
                  turning_radius=10.0, step_size=0.1, max_length=50):
        traj_plan = get_trajectory(start, goal, start_speed, max_speed,
                                   turning_radius, step_size, max_length)
        traj_viz = traj_plan_to_nav(traj_plan)
        self.pub_traj_plan.publish(traj_plan)
        self.pub_traj_viz.publish(traj_viz)
        return

    def _kill_launchfile(self):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('toycar_restart.py')
    args = parser.parse_args()
    try:
        mon = ToyCarEpisodeMonitor()
        mon.spin()
    except rospy.ROSInterruptException:
        pass