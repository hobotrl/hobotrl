# use this node to start/shutdown/restart another node/launch file.
# Author: Michael9792
# Functionality: restart roslaunch file. Can be used to open & close launch file repeatedly.
# When the car stops, it will relaunch everything.
# Usage: 
# 1. set your launch file's name to 'self.process_name' variable
# 2. start roscore in a terminal
# 3. in shell: python rviz_restart.py

import time
import subprocess
import rospy
import rospkg
import numpy as np
from numpy import linalg as LA
from std_msgs.msg import Int16, Bool
from autodrive_msgs.msg import CarStatus
from collections import deque


class restart_ros_launch:
    def __init__(self):
        self.launch_list = list()
        self.socket2 = ""
        self.last_pos = deque(maxlen=200) # Car status is 50Hz, so when car stops moving 4 secs, treat it as stop.
        self.destination = np.zeros([1,3])
        rospack = rospkg.RosPack()  # get an instance of RosPack with the default search paths 
        self.is_running = False
        self.process_name = ['roslaunch', 'planning', 'honda_S5-1.launch']

        rospy.init_node('restart_launch_fle')
        self.heart_beat = rospy.Publisher("/rl/is_running", Bool, latch=True)

    def terminate(self):
        print "========================"
        print "========================"
        print "Publish heart beat False!"
        print "========================"
        print "========================"
        self.heart_beat.publish(False)

        if len(self.launch_list) is 0:
            print("no process to terminate")
        else:
            rospy.loginfo("now shut down launch file")
            self.launch_list[0].terminate()
            self.launch_list[0].wait()
            self.launch_list = []
            print("Shutdown!")

    def restart_callback(self):
        # restart launch file
        rosrun = subprocess.Popen(self.process_name)
        self.launch_list.append(rosrun)
        print("restart launch file finished!")

        print "========================"
        print "========================"
        print "Publish heart beat True!"
        print "========================"
        print "========================"
        self.heart_beat.publish(True)

    def car_out_of_lane_callback(self, data):
        if data.data is 1:
            rospy.logwarn("Car out of Lane!(From error msg)")
              
            self.terminate()
                               
    def car_not_move_callback(self, data):
        # NEED TO BE DONE
        # The car stops or go with a very slow speed, 
        # doesn't mean the car arrived at the destination.
        # maybe we can set a circle near destination, when it reaches that range, we can restart the process 
        # rospy.loginfo("0 is "+ str(self.last_pos[0]))
        # rospy.loginfo("-1 is "+str(self.last_pos[-1]))
        self.last_pos.append(np.array([data.position.x, data.position.y, data.position.z]))
        if len(self.last_pos) is 200 and LA.norm(self.last_pos[0] - self.last_pos[-1])<0.1:
            rospy.logwarn("The car stops moving!")
            self.last_pos.clear() 
            self.terminate()

    def sender(self):
        rospy.Subscriber('/error/type', Int16, self.car_out_of_lane_callback)
        rospy.Subscriber('/car/status', CarStatus, self.car_not_move_callback)
        rospy.Subscriber('rl/simulator_restart', Bool, self.restart_callback)

        rosrun = subprocess.Popen(self.process_name)
        self.launch_list.append(rosrun)
         
        print "========================"
        print "Publish heart beat True!"
        print "========================"
        self.heart_beat.publish(True)

        rospy.spin()

        self.terminate()


if __name__ == '__main__':
    try:
        myobjectx = restart_ros_launch()
        myobjectx.sender()
    except rospy.ROSInterruptException:
        pass
