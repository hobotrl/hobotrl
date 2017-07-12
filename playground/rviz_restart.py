# use this node to start/shutdown/restart another node/launch file.
# Author: Michael9792
# Functionality: restart roslaunch file. Can be used to open & close launch file repeatedly.
# When the car stops, it will relaunch everything.
# Usage: 
# 1. put this file in a <your_package_name>/scripts file path
# 2. in shell: 
# 	cd <your_package_name>/scripts
# 	sudo chmod +x rviz_restart.py
# 3. start roscore in a terminal
# 4. in shell: rosrun <your_package_name> rviz_restart.py


import roslaunch
import time
import rosnode
import rospy
import zmq
import sys
import rospkg
import numpy as np
from numpy import linalg as LA
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from autodrive_msgs.msg import CarStatus
from collections import deque

class MyClass:
    def __init__(self):
        self.launch_list = list()
        self.socket2 = ""
        self.last_pos = deque(maxlen=200) # Car status is 50Hz, so when car stops moving 4 secs, treat it as stop.
        self.destination = np.zeros([1,3])
        rospack = rospkg.RosPack()  # get an instance of RosPack with the default search paths 
        self.launch_file = rospack.get_path('planning')+"/launch/honda_S5-1.launch"
        
    def restart(self):
        # restart launch file
        rospy.loginfo("now start launch file again")
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [self.launch_file])
        launch.start()
        self.launch_list[0] = launch

    def car_out_of_lane_callback(self, data):
        if data.data is 1:
            rospy.logwarn("Car out of Lane!(From error msg)")
            rospy.loginfo("now shut down launch file in 10 seconds")
            self.launch_list[0].shutdown()
            time.sleep(2)
                        
            self.restart()
        
    def car_not_move_callback(self, data):
        self.last_pos.append(np.array([data.position.x, data.position.y, data.position.z]))
        # The car stops or go with a very slow speed, 
        # doesn't mean the car arrived at the destination.
        # maybe we can set a circle near destination, when it reaches the circle, we can restart the process 
        # NEED TO BE DONE
        # rospy.loginfo("0 is "+ str(self.last_pos[0]))
        # rospy.loginfo("-1 is "+str(self.last_pos[-1]))
        if len(self.last_pos) is 200 and LA.norm(self.last_pos[0]-self.last_pos[-1])<0.1:
            # we think the car stops moving
            self.last_pos.clear()
            rospy.logwarn("The car stops moving!")
            rospy.loginfo("now shut down launch file in 10 seconds")
            self.launch_list[0].shutdown()
            time.sleep(2)
            
            self.restart()



    def sender(self):
        rospy.init_node('restart_launch_fle')
        rate = rospy.Rate(1) #1Hz
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [self.launch_file])
        self.launch_list.append(launch)
               
        rospy.Subscriber('/error/type', Int16, self.car_out_of_lane_callback)
        rospy.Subscriber('/car/status', CarStatus, self.car_not_move_callback)
        launch.start()

        while not rospy.is_shutdown():
            rate.sleep()

        self.launch_list[0].shutdown()


if __name__ == '__main__':
    try:
        myobjectx = MyClass()
        myobjectx.sender()
    except rospy.ROSInterruptException:
        pass
