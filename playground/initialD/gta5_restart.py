#!/usr/bin/env python
# use this node to start another node/launch file.


import time
import signal
import subprocess
import os
import socket
import roslaunch
import rosnode
import rospy
import zmq
import rospkg
from timer import Timer
import numpy as np
import argparse
from numpy import linalg as LA
from std_msgs.msg import Int16, Bool
from gta5_interface.msg import gta5_control
from collections import deque
from datetime import datetime
import traceback

# use this to start another node. Take talker.py as an example.
# def sender():
#     rospy.init_node('just_a_test')
#     rate = rospy.Rate(5) #50Hz
#     package = 'gta5_interface'
#     executable = 'talker.py'
#     node = roslaunch.core.Node(package, executable)

#     launch = roslaunch.scriptapi.ROSLaunch()
#     launch.start()

#     process = launch.launch(node)
#     while not rospy.is_shutdown():
#         rate.sleep()
#         rospy.loginfo(str(process.is_alive()))
    
#     process.stop()

# use this to start another launch file.



class MyClass:
    def __init__(self, portnumber, roadindex, recordlog):
        self.process_list = list()
        self.process_names = [
            ['roslaunch', 'gta5_interface', 'simulataneous_control.launch'],
        ]

        # ROS node
        rospy.init_node('GTA5LaunchFileRestarter')

        self.socket2 = ""
        self.car_pos_deque_maxlen = 200
        self.last_pos = deque(maxlen=self.car_pos_deque_maxlen) # gta5_carstatus is 50Hz, so when car stops moving 4 secs, treat it as stop.
        self.restart_times = 0
        self.car_out_of_lane_times = 0
        self.car_hit_obs_times = 0
        self.car_not_move_times = 0
        self.car_cannot_find_current_station_times = 0
        self.port_number = portnumber
        self.road_index = 1 # default road index
        try:
            self.road_index = int(roadindex)            
        except ValueError:
            #Handle the exception
            print 'your input is not a valid integer, now set default road to 1'
        
        rospy.set_param("/port_id", self.port_number)
        auxiliaryportID = self.port_number+1
        context2 = zmq.Context()
        self.socket2 = context2.socket(zmq.REQ)

        try:
            self.socket2.connect("tcp://10.31.40.223:%s" % str(auxiliaryportID))
        except zmq.error.ZMQError as e:
            rospy.logfatal("Please Connect Wi-Fi!")
            sys.exit()
        except Exception as e:
            print e.message
            traceback.print_exc()

        self.record_log = recordlog
        self.destination = np.zeros([1,3])
        rospack = rospkg.RosPack()  # get an instance of RosPack with the default search paths 
        
        # about data management
        if self.record_log is True:
            str_time=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            dirname = rospack.get_path('gta5_interface')+"/loop_log/"
            self.datalog = open(dirname+str_time+".txt","w+")
            self.datalog.write("secs nsecs event\n")

        # set rosparam
        if self.road_index is 1:
            rospy.set_param('/route', '52,53,37,38,35,36,0,46,44,59,54,69,60,75,74,84,76,79,61')
            rospy.set_param('/map/filename',
                            '/home/lewis/Projects/gta_maps/gta5backup/test_alwaysupdate2.xodr')
        elif self.road_index is 2:
            rospy.set_param('/route', '3,5,1,15,10,14,12,17,16,28,27,45,39')
            rospy.set_param('/map/filename', '/home/lewis/Projects/gta_maps/a-lane2/update.xodr')
        else:
            rospy.logfatal("You didn't set right lane index! Program stop!")

        # Simulator states
        self.is_running = False
        self.last_pos = deque(maxlen=1000) # Approximately 20 secs @ 50Hz
        self.last_on_opposite_path = 1

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
            lambda *args: self.heartbeat_pub.publish(self.is_running)
        )

        # periodic opposite path signal
        Timer(
            rospy.Duration(1/20.0),
            lambda *args: self.opposite_path_pub.publish(self.last_on_opposite_path)
        )

        # subscribers
        rospy.Subscriber('/error/type', Int16, self.car_out_of_lane_callback)
        rospy.Subscriber('gta5_carstatus_publisher', gta5_control, self.car_not_move_callback)
        rospy.Subscriber('/rl/simulator_restart', Bool, self.restart_callback)
        rospy.Subscriber('/rl/on_grass', Int16, self.car_out_of_lane_callback) #can not get it
        rospy.Subscriber('/rl/on_opposite_path', Int16, self.__assign_last_op) #can not get it

        # I have following topics can be added to 'reward'
        rospy.Subscriber('gta5_carstatus_publisher', gta5_control, self.car_hit_obs_callback)
        rospy.Subscriber('/error/type_cannotfindstation', Int16, self.car_cannot_find_current_station_callback)
        
    def __assign_last_op(self, data):           
        self.last_on_opposite_path = data.data

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

    def restart_callback(self, data):
        print "[rviz_restart.restart]: restart callback with {}".format(data.data)
        if data.data==False:
            print "[rviz_restart.restart]: mere termination requested."
            self.terminate()
            print "[rviz_restart.restart]: termination finished."
            self.is_running_pub.publish(False)
            return

        # reset start position
        # make car in GTA return to its original place
        self.socket2.send("reset,pos,"+str(self.road_index)+",""")
        msg1 = self.socket2.recv()
        time.sleep(0.1)
        self.socket2.send("control,loop,1,""")
        msg1 = self.socket2.recv()

        # restart launch file
        for name in self.process_names:
            p = subprocess.Popen(name)
            self.process_list.append(p)

        self.last_pos.clear()
        self.restart_times+=1

        print("[rviz_restart.restart]: restarted launch file!")

        self.is_running = True

        print "[rviz_restart.restart]: publish heartbeat=True!"
        self.is_running_pub.publish(True)

    def car_out_of_lane_callback(self, data):
        """Various reasons car is out of lane. (On grass for one)"""
        if abs(int(data.data)-1)<0.001:
            rospy.logwarn("Car out of Lane! (on grass)")
            self.terminate()

            self.car_out_of_lane_times+=1
            if self.record_log is True:
                self.datalog.write(str(rospy.get_rostime().secs) + " " + str(rospy.get_rostime().nsecs) + " " + "car out of lane\n")
       
    def car_hit_obs_callback(self, data):
        if data.is_collision is True:
            rospy.logwarn("You hit an obstacle !(From GTA feedback)")
            rospy.loginfo("now shut down launch file")
            self.terminate()

            self.car_hit_obs_times+=1
            if self.record_log is True:
                self.datalog.write(str(rospy.get_rostime().secs) + " " + str(rospy.get_rostime().nsecs) + " " + "car hits an obstacle\n")


    def car_not_move_callback(self, data):
        self.last_pos.append(np.array([data.car_pos.x, data.car_pos.y, data.car_pos.z]))
        # The car stops or go with a very slow speed, 
        # doesn't mean the car arrived at the destination.
        # maybe we can set a circle near destination, when it reaches the circle, we can restart the process 
        if (len(self.last_pos)==self.last_pos.maxlen and
            LA.norm(self.last_pos[0] - self.last_pos[-1])<0.1):
            rospy.logwarn("The car stops moving!")
            self.last_pos.clear()
            self.terminate()

            self.car_not_move_times+=1
            if self.record_log is True:
                self.datalog.write(str(rospy.get_rostime().secs) + " " + str(rospy.get_rostime().nsecs) + " " + "stops moving\n")

    def car_cannot_find_current_station_callback(self, data):
        if data.data is 1:
            rospy.logwarn("Can not find current station! (From error msg)")
            rospy.loginfo("now shut down launch file")
            self.terminate()

            self.car_cannot_find_current_station_times+=1
            if self.record_log is True:
                self.datalog.write(str(rospy.get_rostime().secs) + " " + str(rospy.get_rostime().nsecs) + " " + "can't find current station\n")


    def sender(self):      

        rospy.spin()
        self.terminate()

        if self.record_log is True:
            self.datalog.write("----------------------------------\n\n")
            self.datalog.write("             Summary              \n\n")
            self.datalog.write("----------------------------------\n\n")
            self.datalog.write("\nTotal restart                      times is "+str(self.restart_times))
            self.datalog.write("\nTotal car goes out of lane         times is "+str(self.car_out_of_lane_times))
            self.datalog.write("\nTotal car hit obstacle             times is "+str(self.car_hit_obs_times))
            self.datalog.write("\nTotal car not move                 times is "+str(self.car_not_move_times))
            self.datalog.write("\nTotal can not find current station times is "+str(self.car_cannot_find_current_station_times))
            self.datalog.close()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--port_number", help="input GTA port number here", type=int)
        parser.add_argument("road_index", help="input road number here, 1 is longer and 2 is a loop", type=str)
        parser.add_argument("--log", help="if you want to log all the process", action="store_true")

        args = parser.parse_args()
        record_log = True if args.log else False
        args.port_number = args.port_number if args.port_number else 3389

        myobjectx = MyClass(args.port_number, args.road_index, record_log)
        myobjectx.sender()
    except rospy.ROSInterruptException:
        pass
