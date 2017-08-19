import os
import signal
import time
import sys
import traceback
sys.path.append('../../..')
sys.path.append('..')

from ros_environments import DrivingSimulatorEnvServer as DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage

env = DrivingSimulatorEnv('22224')

try:
    env.start()
    env.join()
except Exception as e:
    print e.message
finally:
    print "Tidying up..."
    env.exit()
    # kill orphaned monitor daemon process
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


