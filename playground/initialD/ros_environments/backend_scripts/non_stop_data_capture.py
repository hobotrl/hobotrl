import argparse
import traceback
# ROS py
import rospy
# Message types
from std_msgs.msg import Char, Int16
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Path
from autodrive_msgs.msg import CarStatus
# OpenCV related
import cv2
from cv_bridge import CvBridge, CvBridgeError
# Utils
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
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
    global n_latest
    global dec_file
    global status_file
    global new_ep
    while locked:
        print 'Global vars locked!'
        time.sleep(0.1)
    locked = True
    if new_ep:
        if ep_counter%video_every==0:
            if video_wrt is not None:
                video_wrt.release()
            video_wrt = cv2.VideoWriter(
                    video_file_fmt.format(ep_counter),
                    fourcc, 50.0, (320, 320))
            if video_info_file is not None:
                video_info_file.close()
            video_info_file = open(
                video_info_file_fmt.format(ep_counter), 'w')
        video_wrt_latest = cv2.VideoWriter(
            video_file_name_latest.format(ep_counter%n_latest),
            fourcc, 50.0, (320, 320))
        if dec_file is not None:
            dec_file.close()
        dec_file = open(
            dec_file_fmt.format(ep_counter), 'w')
        if status_file is not None:
            status_file.close()
        status_file = open(
            status_file_fmt.format(ep_counter), 'w'
        )
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
    img = brg.compressed_imgmsg_to_cv2(data, 'bgr8')
    img = imresize(img, (320, 320))
    if not new_ep:
        if video_wrt is not None and video_info_file is not None:
            video_wrt.write(img)
            #print "[{}: Video    @ rostime {:.3f}s] seq # {}, delay {}ms".format(
            #    ep_counter, now.to_nsec()/1e9, seq, (now.to_nsec()-ts.to_nsec())/1e6)
            video_info_file.write("{}, {}, {}, {}\n".format(
                seq, ts.to_nsec(), now.to_nsec(), time.time()))
        if video_wrt_latest is not None:
            video_wrt_latest.write(img)

def decision_callback(data):
    global ep_counter
    now = rospy.get_rostime()
    init_global_var(now)
    # print "[{}: Decision @ rostime {:.3f}s] content {}".format(
    #    ep_counter, now.to_nsec()/1e9, str(data.data))
    if not new_ep and dec_file is not None:
        dec_file.write("{}, {}, {}\n".format(
            now.to_nsec(), time.time(), data.data))

def status_callback(data):
    now = rospy.get_rostime()
    # init_global_var(now)
    try:
        x, y = data.position.x, data.position.y
        # dgt = int(now.to_sec()*100)%100
        # if dgt in (0, 1, 24, 25, 49, 50, 74, 75):
        #if True:
        #    print "[{}: Status   @ rostime {:.3f}s] x {}, y {}".format(
        #       ep_counter, now.to_nsec()/1e9, str(x), str(y)
        #   )
        if status_file is not None:
            status_file.write("{}, {}, {}, {}\n".format(
                now.to_nsec(), time.time(), x, y))
    except:
        print "[{}: Status   @ rostime {:.3f}s] status data error!".format(
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
    video_file_name_latest = folder+'video_ep_latest_{}.mp4'
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
    # video_every = 1
    # n_latest = 10
    locked = False
    time_elapsed = 1e4*1e9  # 1e4 seconds 
    new_ep = True

    parser = argparse.ArgumentParser()
    parser.add_argument("n_ep", help="Num of episode to recored", type=int)
    parser.add_argument("video_every", nargs="?",
                        help="Record video every # of episodes",
                        type=int, default=1)
    parser.add_argument("n_latest", nargs="?",
                        help=("Num of latest video files to process on a "
                              "rolling basis"),
                        type=int, default=10)
    args = parser.parse_args()
    print "[VideoRecorder]: Parsed n_ep is {}".format(args.n_ep)
    if args.n_ep > 0:
        ep_counter = args.n_ep
    video_every = args.video_every
    n_latest = args.n_latest

    try:
        print "[VideoRecorder]: recording episode {}".format(ep_counter)
        rospy.init_node('training_data_capture')
        # if ep_counter%video_every==0:
        rospy.Subscriber("/training/image/compressed",
                         CompressedImage, bdview_callback)
        rospy.Subscriber("/autoDrive_KeyboardMode", Char, decision_callback)
        rospy.Subscriber("/car/status", CarStatus, status_callback)
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
    print "All files closed properlly for episode {}.".format(ep_counter)
