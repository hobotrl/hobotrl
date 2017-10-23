"""Toy car reset helper functions.
"""
import numpy as np
from dubins import path_sample as dubin_path_sample
import rospy
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from nav_msgs.msg import Path
from autodrive_msgs.msg import PlanningPoint, PlanningTraj

def get_trajectory(start, goal, start_speed, max_speed,
                   turning_radius=10.0, step_size=0.1, max_length=50):
    qs = sample_points(start, goal, turning_radius, step_size)
    qs = qs[:int(max_length/step_size)]
    speeds = sample_speeds(qs, start_speed, max_speed)
    traj = [
        PlanningPoint(**{'x': q[0], 'y': q[1], 'theta': q[2], 'speed': v})
        for q, v in zip(qs, speeds)
    ]
    time = rospy.rostime.get_rostime()
    header = Header(*[0, time, 'map'])
    return PlanningTraj(**{'header': header, 'trajectory': traj})

def sample_points(start, goal, rho, step_size):
    return dubin_path_sample(start, goal, rho, step_size)[0]

def sample_speeds(qs, start_speed, max_speed):
    speeds = np.ones(len(qs)) * max_speed  # array of speeds
    len_acc = len(speeds)/3  # acceleration length
    speeds[:len_acc] = np.linspace(start_speed, max_speed, len_acc)
    speeds[-len_acc:] = np.linspace(speeds[-len_acc], 0, len_acc)
    return speeds

def traj_plan_to_nav(plan_traj):
    poses = []
    time = rospy.rostime.get_rostime()
    for i, p in enumerate(plan_traj.trajectory):
        position = Point(*[p.x, p.y, p.z])
        orientation = Quaternion(
            *quaternion_from_euler(0, 0, p.theta).tolist()
        )
        pose = Pose(*[position, orientation])
        header = Header(*[i, time, 'map'])
        poses.append(PoseStamped(**{
            'header': header, 'pose': pose
        }))
    header = Header(*[0, time, 'map'])
    return Path(**{'header': header, 'poses': poses})




