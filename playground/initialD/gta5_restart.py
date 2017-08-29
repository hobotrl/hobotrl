#!/usr/bin/env python
# use this node to start another node/launch file.

import time
import signal
import subprocess
import rospy
import socket
import rosgraph
import rospkg
import zmq
import numpy as np
from numpy import linalg as LA
from std_msgs.msg import Char, Int16, Bool
from autodrive_msgs.msg import CarStatus
from gta5_interface.msg import gta5_control
from collections import deque
from ros_utils.timer import Timer
import argparse

class MyClass:
    def __init__(self, portnumber, roadindex, ipaddress):
        # process related
        self.process_list = list()
        self.process_names = [
            ['roslaunch', 'gta5_interface', 'simulataneous_control.launch'],
            # ['python', '/home/lewis/Projects/hobotrl/playground/initialD/gazebo_rl_reward.py']
        ]
        # Simulator states
        self.is_running = False
        self.last_pos = deque(maxlen=1000) # Approximately 20 secs @ 50Hz
        self.last_pos_tmp_not_move = deque(maxlen=200) # 4s not move, send '1' and 'g'
        self.last_on_opposite_path = 1
        self.port_number = portnumber
        self.ip_address = ipaddress
        self.road_index = roadindex
        rospack = rospkg.RosPack()  # get an instance of RosPack with the default search paths 

        # make sure roscore is running
        try:
            rosgraph.Master('/rostopic').getPid()
        except socket.error:
            print("\nUnable to communicate with master!\n")
            traceback.print_exc()

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
            "/rl/last_on_opposite_path", Int16, queue_size=10, latch=True
        )
        # periodic heartbeat
        Timer(
            rospy.Duration(1/20.0),
            lambda *args: self.heartbeat_pub.publish(self.is_running))
        # periodic opposite path signal
        Timer(
            rospy.Duration(1/20.0),
            lambda *args: self.opposite_path_pub.publish(self.last_on_opposite_path)
        )

        # set rosparam
        if self.road_index is 1:
            rospy.set_param('/route', '52,53,37,38,35,36,0,46,44,59,54,69,60,75,74,84,76,79,61')
            rospy.set_param('/map/filename', rospack.get_path('gta5_interface')+'/data/lane1.xodr')
        elif self.road_index is 2:
            rospy.set_param('/route', '3,5,1,15,10,14,12,17,16,28,27,45,39')
            rospy.set_param('/map/filename', rospack.get_path('gta5_interface')+'/data/lane2.xodr')
        else:
            rospy.logfatal("You didn't set right lane index! Program stop!")

        if self.ip_address:
            rospy.set_param('/gta/ip_address', self.ip_address)
        else:
            rospy.logfatal("you didn't set GTA ip address! Program stop!")

        rospy.set_param("/port_id", self.port_number)
        auxiliaryportID = self.port_number+1
        context2 = zmq.Context()
        self.socket2 = context2.socket(zmq.REQ)

        try:
            self.socket2.connect("tcp://%s:%s" % (str(self.ip_address), str(auxiliaryportID)))
        except zmq.error.ZMQError as e:
            rospy.logfatal("Please Connect Ethernet!")
            sys.exit()

        # subscribers
        rospy.Subscriber('/error/type', Int16, self.car_out_of_lane_callback)
        rospy.Subscriber('/car/status', CarStatus, self.car_not_move_callback)
        rospy.Subscriber('gta5_carstatus_publisher', gta5_control, self.car_not_move_temp_callback)
        rospy.Subscriber('/rl/simulator_restart', Bool, self.restart_callback)
        rospy.Subscriber('/rl/on_grass', Int16, self.car_out_of_lane_callback)
        rospy.Subscriber('/rl/on_opposite_path', Int16, self.__assign_last_op)
        rospy.Subscriber('gta5_carstatus_publisher', gta5_control, self.car_hit_obs_callback)
        rospy.Subscriber('/error/type_cannotfindstation', Int16, self.car_cannot_find_current_station_callback)
        self.pub_temp_restart = rospy.Publisher('/autoDrive_KeyboardMode', Char, queue_size=10)

    def __assign_last_op(self, data):
        self.last_on_opposite_path = data.data

    def terminate(self):
        # flush heartbeat = False for 1 sec
        self.is_running = False
        secs = 1
        while secs != 0:
            print "[gta5_restart.terminate]: Shutdown simulator nodes in {} secs".format(secs)
            secs -= 1
            time.sleep(1.0)

        # shutdown simulator node
        if len(self.process_list) is 0:
            print("[gta5_restart.terminate]: no process to terminate")
        else:
            rospy.loginfo("now shut down launch file")
            for p in self.process_list:
                # p.terminate()
                # p.kill()
                p.send_signal(signal.SIGINT)
                while p.poll() is None:
                    print (
                        "[gta5_restart.terminate]: Simulator proc {} termination in progress..."
                    ).format(p.pid)
                    time.sleep(1.0)
                print (
                    "[gta5_restart.terminate]: Simulator proc {} terminated with exit code {}"
                ).format(p.pid, p.returncode)
            self.process_list = []
            print("Done!")

        # signal env node shutdown
        print "[gta5_restart.terminate]: publish heartbeat=False!"
        self.is_running_pub.publish(False)

    def car_cannot_find_current_station_callback(self, data):
        if data.data is 1:
            rospy.logwarn("Can not find current station! (From error msg)")
            rospy.loginfo("now shut down launch file")
            self.terminate()

    def restart_callback(self, data):
        print "[gta5_restart.restart]: restart callback with {}".format(data.data)
        if data.data==False:
            print "[gta5_restart.restart]: mere termination requested."
            self.terminate()
            print "[gta5_restart.restart]: termination finished."
            self.is_running_pub.publish(False)
            return

        # reset start position
        self.socket2.send("reset,pos,"+str(self.road_index)+",""")
        msg1 = self.socket2.recv()
        time.sleep(0.1)
        self.socket2.send("control,loop,1,""")
        msg1 = self.socket2.recv()

        # restart launch file
        for name in self.process_names:
            p = subprocess.Popen(name)
            self.process_list.append(p)
        print("[gta5_restart.restart]: restarted launch file!")

        self.is_running = True

        print "[gta5_restart.restart]: publish heartbeat=True!"
        self.is_running_pub.publish(True)

    def car_out_of_lane_callback(self, data):
        """Various reasons car is out of lane. (On grass for one)"""
        if abs(int(data.data)-1)<0.001:
            rospy.logwarn("Car out of Lane! (on grass)")
            self.terminate()

    def car_not_move_callback(self, data):
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

    def car_hit_obs_callback(self, data):
        if data.is_collision is True:
            rospy.logwarn("You hit an obstacle !(From GTA feedback)")
            rospy.loginfo("now shut down launch file")
            self.terminate()

    def sender(self):
        rospy.spin()
        self.terminate()

    def car_not_move_temp_callback(self, data):
        self.last_pos_tmp_not_move.append(np.array([data.car_pos.x, data.car_pos.y, data.car_pos.z]))
        # The car stops or go with a very slow speed, 
        # doesn't mean the car arrived at the destination.
        # maybe we can set a circle near destination, when it reaches the circle, we can restart the process 
        # if len(self.last_pos) is 200 and (self.last_pos[0] == self.last_pos[-1]).all():
        # NEED TO BE DONE
        if len(self.last_pos_tmp_not_move) == self.last_pos_tmp_not_move.maxlen and 
        LA.norm(self.last_pos_tmp_not_move[0] - self.last_pos_tmp_not_move[-1])<0.1:
            # we think the car stops moving
            # send '1' 'g' now
            r = rospy.Rate(2) # 10hz
            self.pub_temp_restart.publish(ord('1'))
            r.sleep()
            self.pub_temp_restart.publish(ord('g'))
            r.sleep()
            self.pub_temp_restart.publish(ord('1'))
            r.sleep()
            self.pub_temp_restart.publish(ord('g'))
            r.sleep()

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--port_number", help="input GTA port number here", type=int)
        parser.add_argument("road_index", help="input road number here, 1 is longer and 2 is a loop", type=int)
        parser.add_argument("--log", help="if you want to log all the process", action="store_true")
        parser.add_argument("--ip", help="input GTA PC's ip here", type=str)

        args = parser.parse_args()
        record_log = True if args.log else False
        args.port_number = args.port_number if args.port_number else 10000
        args.ip = args.ip if args.ip else "10.31.40.215"

        myobjectx = MyClass(args.port_number, args.road_index, args.ip)
        myobjectx.sender()
    except rospy.ROSInterruptException:
        pass
