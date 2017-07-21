from ros_environments import DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Float32
from sensor_msgs.msg import Image

env = DrivingSimulatorEnv(
    [('/training/image', Image)],
    [('/rl/has_obstacle_nearby', Bool),
     ('/rl/distance_to_longestpath', Float32)],
    [('/rl/action/test', Float32)],
    rate_action=1,
    buffer_sizes={'observation': 10, 'reward': 10, 'action': 10}
)
try:
    env.step((1.0, ))
    rospy.spin()
except rospy.ROSInterruptException:
    pass
