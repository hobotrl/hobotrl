"""Reward calculator and publisher.

This scrips calculates components of RL reward based on ROS topics published by
the simulator. The calculated reward components are published again as ROS
topics.

Reward components:
    1. '/rl/has_obstacle_nearby': whether there's obstacle cars near by.
    2. '/rl/distance_to_longestpath': perpendicular distance to the closest
        point from ego car to '/path/longest'.
    3. '/rl/car_velocity': car speed.
    4. '/rl/car_velocity_front': car speed along longest path.
    5. '/rl/last_on_opposite_path': whether car is on opposite lane.
    6. '/rl/on_pedestrian': whether ego car is on pedestrian lane.
    7. '/rl/obs_factor': directional obstacle risk factor.
:author: Gang XU, Jingchu LIU
:date: 2017-09-06
"""
import os
from os.path import dirname, realpath
import sys
import zmq
import numpy as np
from numpy import linalg as LA
import cv2
import matplotlib.pyplot as plt
# ROS
import rospy
import rospkg
from tf.transformations import euler_from_quaternion
from autodrive_msgs.msg import Control, Obstacles, CarStatus
from std_msgs.msg import Bool, Int16, Float32
from nav_msgs.msg import Path
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from ros_environments.utils.timer import Timer

class RewardFunction:
    def __init__(self):
        """Initialization."""
        # Latched status info
        # 1. x,y,z coord. of ego car 
        self.car_pos = np.zeros((3,))
        # 2. roll, pitch, yaw of ego car
        self.car_euler = np.zeros((3,))
        # 3. distance to longest_path
        self.last_dist_longestpath = 0.0
        self.last_yaw_longestpath = 0.0
        # 4. obstacle related
        # distance threshold to be considered as collision
        # note: distance is calculated with geometric center, so has to take
        # the diameter of objects. Diameters of car models are roughly 5
        # meters.
        self.detect_obstacle_range = 5 + 1
        self.min_obs_dist = 0.0
        self.obs_risk = 0.0
        # 5. opposite path
        self.last_on_opp = 1

        self.brg = CvBridge()

        # ROS related
        rospy.init_node('rl_reward_fcn')
        self.pub_nearest_obs = rospy.Publisher(
            '/rl/has_obstacle_nearby', Bool, queue_size=100)
        self.pub_obs_risk = rospy.Publisher(
            '/rl/obs_factor', Float32, queue_size=100)
        self.pub_closest_dist_longestpath = rospy.Publisher(
            '/rl/distance_to_longestpath', Float32, queue_size=100)
        self.pub_car_velocity = rospy.Publisher(
            '/rl/car_velocity', Float32, queue_size=100)
        self.pub_car_velocity_f = rospy.Publisher(
            '/rl/car_velocity_front', Float32, queue_size=100)
        self.pub_on_opp = rospy.Publisher(
            "/rl/last_on_opposite_path", Int16, queue_size=100)
        self.pub_on_pedestrian = rospy.Publisher(
            '/rl/on_pedestrian', Bool, queue_size=100)
        self.pub_on_pedestrian_tilt = rospy.Publisher(
            '/rl/on_pedestrian_tilt', Bool, queue_size=100)
        rospy.Subscriber('/path/longest', Path, self.longest_path_callback)
        rospy.Subscriber('/obstacles', Obstacles, self.obstacles_callback)
        rospy.Subscriber('/car/status', CarStatus, self.car_status_callback)
        rospy.Subscriber(
            '/training/image/compressed', CompressedImage, self.trn_image_callback)
        rospy.Subscriber('/rl/on_opposite_path', Int16, self.on_opp_callback)
        Timer(
            rospy.Duration(1/20.0),
            lambda *args: self.pub_on_opp.publish(self.last_on_opp))

    def car_status_callback(self, data):
        """Callback for '/car/status'."""
        self.car_pos = np.array(
            [data.position.x, data.position.y, data.position.z])
        self.car_euler = euler_from_quaternion(
            (data.orientation.x, data.orientation.y,
             data.orientation.z, data.orientation.w))
        speed = data.speed
        speed_f = np.abs(speed * np.dot(
            (np.cos(self.car_euler[2]), np.sin(self.car_euler[2])),
            (np.cos(self.last_yaw_longestpath), np.sin(self.last_yaw_longestpath))
        ))
        self.pub_car_velocity.publish(speed)
        self.pub_car_velocity_f.publish(speed_f)

    def longest_path_callback(self, data):
        """Callback for '/path/longest'

        Calculate the minimum perpendicular distance from ego car to the
        longest path.

        SVD is used to find the approximate tangent direction around the
        closest point.
        """
        min_idx = self.find_minimum_distance(data.poses)  # closest point index
        if min_idx is None:
            return
        else:
            # extract 20 points along the closest point
            # use z position of ego car since z displacement doesn't matter
            # truncate if encounter head or tail
            path_points = np.array([
                (pose.pose.position.x, pose.pose.position.y, self.car_pos[2])
                for pose in data.poses[max(min_idx-10, 0):min_idx+10]]
            )
            # use svd to find the approximate tangent direction of longest path
            approx_dir = np.linalg.svd(path_points-np.mean(path_points,axis=0))[2][0]
            self.last_yaw_longestpath = np.arctan2(approx_dir[1], approx_dir[0])
            # perpendicular distance is then the norm of vector
            #   (car_pos - pos_point) x approx_dir, x is cross product
            self.last_dist_longestpath = np.linalg.norm(
                np.cross(path_points[0,:] - self.car_pos, approx_dir)
            )
            # publish
            self.pub_closest_dist_longestpath.publish(self.last_dist_longestpath)

    def obstacles_callback(self, data):
        """Callback for '/obstacles'.

        Calculates directional obstacle risk factor and indicator for
        obstacles.
        """
        obs_pos = [(obs.ObsPosition.x, obs.ObsPosition.y, obs.ObsPosition.z)
                   for obs in data.obs]
        obs_yaw = np.array([obs.ObsTheta for obs in data.obs])
        if len(obs_pos)==0:
            self.obs_risk = 0.0
            self.min_obs_dist = self.detect_obstacle_range + 100.0
        else:
            disp_vec = np.array(obs_pos) - self.car_pos  # displacement
            dist_obs = np.linalg.norm(disp_vec, axis=1)  # obstacle distance
            # ego heading unit vector
            ego_hdg = (np.cos(self.car_euler[2]), np.sin(self.car_euler[2]), 0)
            # cosine of ego heading and obs displacment
            obs_cosine = np.dot(disp_vec, ego_hdg)/dist_obs
            # angle of obs displacement w.r.t ego heading
            obs_angle = np.arccos(obs_cosine)
            # raised cosine, 1.0 within a narrow angle ahead, quickly rolloff
            # to 0.0 as angle increases 
            obs_rcos = self.raised_cosine(obs_angle, np.pi/24, np.pi/48)
            # distance risk is Laplacian normalized by detection rangei
            risk_dist = np.exp(-0.1*(dist_obs-self.detect_obstacle_range))
            # relative angle between headings of ego car and obs car
            # shifted by pi
            rel_angle = self.car_euler[2] - obs_yaw + np.pi
            rel_angle = (rel_angle + np.pi) % (2*np.pi) - np.pi
            collide_rcos = self.raised_cosine(rel_angle, np.pi/24, np.pi/48)
            # total directional obs risk is distance risk multiplied by
            # raised-cosied directional weight.
            self.obs_risk = np.sum(
                risk_dist * (obs_rcos+0.1) * (collide_rcos+0.1)
            )
            if np.isnan(self.obs_risk):
                self.obs_risk = 0.0
            # idx = np.argsort(dist_obs)[::]
            # minimum obs distance
            self.min_obs_dist = min(dist_obs)
        near_obs = True if self.min_obs_dist<self.detect_obstacle_range else False
        self.pub_obs_risk.publish(self.obs_risk)
        self.pub_nearest_obs.publish(near_obs)

    def trn_image_callback(self, data):
        """Callback for '/training/image/*'."""
        img = self.brg.compressed_imgmsg_to_cv2(data, 'rgb8')
        # Pedestrian factor
        #   Since Ped lane is the outmost lane, if we take a patch of image
        #   centered around ego car, then there will a considerable portions
        #   of black pixels, i.e. RGB =(0,0,0).
        patch, sum_sq = self._patch_topdown(img)
        ped_factor = np.sum(np.mean(patch, axis=2)<10.0)/(1.0*sum_sq)
        self.pub_on_pedestrian.publish(ped_factor>0.05)
        patch, sum_sq = self._patch_tilted(img)
        ped_factor = np.sum(np.mean(patch, axis=2)<10.0)/(1.0*sum_sq)
        self.pub_on_pedestrian_tilt.publish(ped_factor>0.05)

    def _patch_topdown(self, img):
        offset_y = int(-375/1400.0*img.shape[0])
        offset_x = int(0/1400.0*img.shape[0])
        low = int(np.floor(650/1400.0*img.shape[0]))
        high = int(np.ceil(750/1400.0*img.shape[0]))
        sum_sq = (high-low)**2
        patch = img[
            (offset_x+low):(offset_x+high),
            (offset_y+low):(offset_y+high), :]
        return patch, sum_sq

    def _patch_tilted(self, img):
        offset_y = int(0/1400.0*img.shape[0])
        offset_x = int(+180/1400.0*img.shape[0])
        low = int(np.floor(650/1400.0*img.shape[0]))
        high = int(np.ceil(750/1400.0*img.shape[0]))
        sum_sq = (high-low)**2
        patch = img[
            (offset_x+low):(offset_x+high),
            (offset_y+low):(offset_y+high), :]
        return patch, sum_sq

    def on_opp_callback(self, data):
        self.last_on_opp = data.data

    def raised_cosine(self, x, offset, rolloff):
        """Raised cosine function.

        A piecewise function defined of (-pi, +pi):
            1.0, if |x| < offset - rolloff
            0.0, if |x| > offset + rolloff
            \frac{1}{2}(1+cos(...)), otherwise
        """
        x = np.abs(x)
        cutoff_1, cutoff_0 = offset-rolloff, offset+rolloff
        ind_1, ind_0= x < cutoff_1, x > cutoff_0
        rcos = 0.5*(1+np.cos((x-cutoff_1)/(2*rolloff)*np.pi))
        return ind_1*np.ones_like(x) + \
               ind_0*np.zeros_like(x) + \
               (1-np.logical_or(ind_1, ind_0))*rcos

    def calc_dist(self, p):
        """Calculate the dist between p and car_position."""
        p = np.array((p.x, p.y, p.z))
        return LA.norm(p - self.car_pos)

    def find_minimum_distance(self, poses):
        dist_list = [self.calc_dist(pose.pose.position) for pose in poses]
        if len(dist_list) is 0:
            idx = None
        else:
            # sort by distance value, break tie with smaller index
            val, idx = min((val, idx) for (idx, val) in enumerate(dist_list))
        return idx

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    print "[rl_reward_function]: inside file."
    try:
        rewardfunc = RewardFunction()
        rewardfunc.spin()
    except rospy.ROSInterruptException:
        pass
    print "[rl_reward_function]: out."
