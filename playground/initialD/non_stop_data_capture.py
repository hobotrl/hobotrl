# ROS py
import rospy
# Message types
from std_msgs.msg import Char, Int16
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
    global video_wrt_latest
    global dec_file
    global status_file
    global new_ep
    while locked:
        print 'Global vars locked!'
        time.sleep(0.1)
    locked = True
    if new_ep:
    # if now.to_nsec() < time_elapsed:
        # ep_counter += 1
        print "New episode {} detected".format(ep_counter)
        if ep_counter%video_every==0:
            if video_wrt is not None:
                video_wrt.release()
            video_wrt = cv2.VideoWriter(
                    video_file_fmt.format(ep_counter),
                    fourcc, 40.0, (1400, 1400))
            if video_info_file is not None:
                video_info_file.close()
            video_info_file = open(
                video_info_file_fmt.format(ep_counter), 'w')
        video_wrt_latest = cv2.VideoWriter(
            video_file_name_latest,
            fourcc, 40.0, (1400, 1400))
        print video_wrt_latest
        if dec_file is not None:
            dec_file.close()
        dec_file = open(
            dec_file_fmt.format(ep_counter), 'w')
        # if status_file is not None:
        #     status_file.close()
        #status_file = open(
        #    status_file_fmt.format(ep_counter), 'w'
        #)
    time_elapsed = now.to_nsec()
    locked = False
    new_ep = False

# call-backs
def bdview_callback(data):
    global ep_counter
    global new_ep
    now = rospy.get_rostime()
    # init_global_var(now)
    seq = data.header.seq
    ts = data.header.stamp
    img = brg.imgmsg_to_cv2(data, 'bgr8')
    if not new_ep:
        if video_wrt is not None and video_info_file is not None:
            video_wrt.write(img)
            #print "[{}: Video    @ rostime {:.3f}s] seq # {}, delay {}ms".format(
            #    ep_counter, now.to_nsec()/1e9, seq, (now.to_nsec()-ts.to_nsec())/1e6)
            video_info_file.write(
                str((seq, ts.to_nsec(), now.to_nsec())) + '\n'
            )
        video_wrt_latest.write(img)

def decision_callback(data):
    global ep_counter
    now = rospy.get_rostime()
    init_global_var(now)
    # print "[{}: Decision @ rostime {:.3f}s] content {}".format(
    #    ep_counter, now.to_nsec()/1e9, str(data.data))
    if not new_ep:
        dec_file.write(str((now.to_nsec(), data.data))+'\n')

def path_callback(data):
    now = rospy.get_rostime()
    # init_global_var(now)
    try:
        position = data.poses[0].pose.position
        # if int(now.to_sec()*10)%20==0:
        #    print "[{}: Path     @ rostime {:.3f}s] x {}, y {}".format(
        #        ep_counter, now.to_nsec()/1e9, str(position.x), str(position.y)
        #    )
        # status_file.write("{}, {}, {}\n".format(now.to_nsec(), position.x, position.y))
    except:
        print "[{}: Path     @ rostime {:.3f}s] path data error!".format(
            ep_counter, now.to_nsec())
        pass

# ======= Main Loop =============
if __name__ == '__main__':
    # check folder
    # ==== set folder here ======
    folder = './experiment/video_files/'
    # ===========================
    if not os.path.isdir(folder):
        os.makedirs(folder)
    # Video codec, writer, & info file
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    video_wrt = None
    video_info_file = None
    video_file_fmt = folder+'video_ep_{}.mp4'
    video_info_file_fmt = folder+'video_info_ep_{}.log'
    video_wrt_latest = None
    video_file_name_latest = folder+'video_ep_latest.mp4'
    # Decision log file
    dec_file = None
    dec_file_fmt = folder+'decisions_ep_{}.log'
    # Status log file 
    status_file = None
    status_file_fmt = folder+'status_ep_{}.log'

    # counters, lock, timers
    files = os.listdir(folder)
    files.sort()
    files = map(lambda name: name.split('_')[-1], files)
    files = map(lambda name: name.split('.'), files)
    def to_int_or_none(name):
        try:
            return int(name[-2])
        except:
            return None
    # files = map(lambda name: int(name[0]), files)
    files = map(to_int_or_none, files)
    files = [f for f in files if f is not None]
    # print files
    # print files
    ep_counter = max(files)+1 if len(files)!=0 else 1
    video_every = 10
    locked = False
    time_elapsed = 1e4*1e9  # 1e4 seconds 
    new_ep = True

    try:
        print "[VideoRecorder]: recording episode {}".format(ep_counter)
        rospy.init_node('training_data_capture')
        # if ep_counter%video_every==0:
        rospy.Subscriber("/training/image", Image, bdview_callback)
        rospy.Subscriber("/autoDrive_KeyboardMode", Char, decision_callback)
        rospy.Subscriber("/path/longest", Path, path_callback)
        rospy.spin()
    finally:
        if video_wrt is not None:
            video_wrt.release()
        if video_wrt_latest is not None:
            video_wrt_latest.release() 
        if video_info_file is not None:
            video_info_file.close()
        if dec_file is not None:
            dec_file.close()
        if status_file is not None:
            status_file.close()
    print "All files closed properlly."
