#!/usr/bin/env python

# this python script send useful data to rl network including speed, obstacle reward and closest distance towards longest path.
# data source is from honda simulator directly

import sys
import zmq
import numpy as np
from numpy import linalg as LA
import cv2
# ROS
import rospy
import rospkg
from autodrive_msgs.msg import Control, Obstacles, CarStatus
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Path
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class RewardFunction:
    def __init__(self):
        rospy.init_node('gazebo_rl_reward_fcn')
        self.car_pos_x = 0.0
        self.car_pos_y = 0.0
        self.min_path_dis = 0.0
        self.detect_obstacle_range = 3
        self.closest_distance = 10000.0 # initializer

        self.brg = CvBridge()

        self.pub_nearest_obs = rospy.Publisher(
            '/rl/has_obstacle_nearby', Bool, queue_size=1000)
        self.pub_closest_distance = rospy.Publisher(
            '/rl/distance_to_longestpath', Float32, queue_size=1000)
        self.pub_car_velocity = rospy.Publisher(
            '/rl/car_velocity', Float32, queue_size=1000)
        self.pub_on_pedestrian = rospy.Publisher(
            '/rl/on_pedestrian', Bool, queue_size=1000)
        rospy.Subscriber('/path/longest', Path, self.calc_nearest_distance_callback)
        rospy.Subscriber('/obstacles', Obstacles, self.calc_nearest_obs_callback)
        rospy.Subscriber('/car/status', CarStatus, self.get_status_callback)
        rospy.Subscriber('/training/image/compressed',
                         CompressedImage, self.trn_image_callback)

    def calc_dist(self, p):
        """Calculate the dist between p and car_position."""
        return LA.norm([
            (p.pose.position.x-self.car_pos_x),
            (p.pose.position.y-self.car_pos_y)])

    def find_minimum_distance(self, params):
        new_list = [self.calc_dist(x) for x in params]
        if len(new_list) is 0:
            idx = None
        else:
            # sort by distance value, break tie with smaller index
            val, idx = min((val, idx) for (idx, val) in enumerate(new_list))
        return idx

    def get_status_callback(self, data):
        self.car_pos_x = data.position.x
        self.car_pos_y = data.position.y
        self.pub_car_velocity.publish(data.speed)

    def calc_nearest_distance_callback(self, data):
        aaa = list()
        min_idx = self.find_minimum_distance(data.poses)
        if min_idx is None:
            return
        else:
            # A,B is two closest points to the car along longest path
            aaa.append(np.array([(data.poses[min_idx].pose.position.x-self.car_pos_x), (data.poses[min_idx].pose.position.y-self.car_pos_y)], dtype=np.float64))
            aaa.append(np.array([(data.poses[min_idx+1].pose.position.x-self.car_pos_x), (data.poses[min_idx+1].pose.position.y-self.car_pos_y)], dtype=np.float64))
            vec_AB = aaa[0] - aaa[1]
            sine_val = np.abs(np.cross(aaa[0], vec_AB))/LA.norm(aaa[0])/LA.norm(vec_AB)
            self.min_path_dis = LA.norm(aaa[0]) * sine_val
            self.pub_closest_distance.publish(self.min_path_dis)
            self.closest_distance = 10000.0

    def calc_nearest_obs_callback(self, data):
        new_list = [LA.norm([i.ObsPosition.x-self.car_pos_x, i.ObsPosition.y-self.car_pos_y]) for i in data.obs]
        self.closest_distance = min(new_list) if not len(new_list)==0 else 10000.0
        near_obs = True if self.closest_distance<self.detect_obstacle_range else False
        self.pub_nearest_obs.publish(near_obs)

    def trn_image_callback(self, data):
        img = self.brg.compressed_imgmsg_to_cv2(data, 'rgb8')
        # Pedestrian factor
        #   Since Ped lane is the outmost lane, if we take a 100x100 slice
        #   centered around ego car, there will be considerable portions of
        #   pixels being (0,0,0) if ego car is on the Ped lane. Thus the sum
        #   lumanation will be lower compared with other cases. 
        low = int(np.floor(650.0/1400.0*img.shape[0]))
        high = int(np.ceil(750.0/1400.0*img.shape[0]))
        sum_sq = (high-low)**2
        ped_factor = np.sum(img[low:high, low:high, :])/(255*3*sum_sq)  # norm by max val
        self.pub_on_pedestrian.publish(ped_factor<0.31)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    print "[gazebo_rl_reward]: inside file."
    try:
        rewardfunc = RewardFunction()
        rewardfunc.spin()
    except rospy.ROSInterruptException:
        pass
    print "[gazebo_rl_reward]: out."
