# ROS py
import rospy
# Message types
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
# OpenCV related
import cv2
from cv_bridge import CvBridge, CvBridgeError
# Utils
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# CvBridge
brg = CvBridge()

# global_var checking fcn
def init_global_var(now):
    global locked
    global ep_counter
    global time_elapsed
    global video_wrt
    global video_info_file
    global dec_file
    global status_file
    while locked:
        print 'Global vars locked!'
        time.sleep(0.1)
    locked = True
    if now.to_nsec() < time_elapsed:
        ep_counter += 1
        print "New episode {} detected".format(ep_counter)
        if video_wrt is not None:
            video_wrt.release()
        video_wrt = cv2.VideoWriter(
                video_file_fmt.format(ep_counter),
                fourcc, 5.0, (1400, 1400)
        )
        if video_info_file is not None:
            video_info_file.close()
        video_info_file = open(
            video_info_file_fmt.format(ep_counter), 'w'
        )
        if dec_file is not None:
            dec_file.close()
        dec_file = open(
            dec_file_fmt.format(ep_counter), 'w'
        )
        if status_file is not None:
            status_file.close()
        status_file = open(
            status_file_fmt.format(ep_counter), 'w'
        )
    time_elapsed = now.to_nsec()
    locked = False

# call-backs
def bdview_callback(data):
    now = rospy.get_rostime()
    init_global_var(now)
    seq = data.header.seq
    ts = data.header.stamp
    img = brg.imgmsg_to_cv2(data, 'bgr8')
    video_wrt.write(img)
    print "[Video    @ rostime {:.3f}s] seq # {}, delay {}ms".format(
        now.to_nsec()/1e9, seq, (now.to_nsec()-ts.to_nsec())/1e6
    )
    video_info_file.write(
        str((seq, ts.to_nsec(), now.to_nsec())) + '\n'
    )

def decision_callback(data):
    now = rospy.get_rostime()
    init_global_var(now)
    print "[Decision @ rostime {:.3f}s] content {}".format(
        now.to_nsec()/1e9, str(data.data)
    )
    dec_file.write(str((now.to_nsec(), data.data))+'\n')

def path_callback(data):
    now = rospy.get_rostime()
    init_global_var(now)
    try:
        position = data.poses[0].pose.position
        print "[Path     @ rostime {:.3f}s] x {}, y {}".format(
            now.to_nsec()/1e9, str(position.x), str(position.y)
        )
        status_file.write("{}, {}, {}\n".format(now.to_nsec(), position.x, position.y))
    except:
        print "[Path     @ rostime {:.3f}s] path data error!".format(
            now.to_nsec()
        )
    
# ======= Main Loop =============
if __name__ == '__main__':
    # check folder
    # ==== set folder here ======
    folder = './trn_honda_J1-1/'
    # ===========================
    if not os.path.isdir(folder):
        os.makedirs(folder)
    else:
        while(True):
            c = raw_input(folder + ' already exist, override? (y/n)')
            if c == 'n':
                print "Exiting..."
                exit()
            elif c == 'y':
                print "Cleaning folder {}...".format(folder),
                shutil.rmtree(folder)
                print 'done!'
                os.makedirs(folder)
                break
            else:
                continue
            break
    # Video codec, writer, & info file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_wrt = None
    video_info_file = None
    video_file_fmt = folder+'video_ep_{}.avi'
    video_info_file_fmt = folder+'video_info_ep_{}.log'
    # Decision log file
    dec_file = None
    dec_file_fmt = folder+'decisions_ep_{}.log'
    # Status log file 
    status_file = None
    status_file_fmt = folder+'status_ep_{}.log'

    # counters, lock, timers
    ep_counter = -1
    locked = False
    time_elapsed = 1e4*1e9  # 1e4 seconds 

    try:
        rospy.init_node('training_data_capture')
        rospy.Subscriber("/training/image", Image, bdview_callback)
        rospy.Subscriber("/decision_result", Int16, decision_callback)
        rospy.Subscriber("/path/longest", Path, path_callback)
        rospy.spin()
    finally:
        video_wrt.release()
        video_info_file.close()
        dec_file.close()
        status_file.close()
        print "All files closed properlly."
