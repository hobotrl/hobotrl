"""Records training_image, decision, and car status topic into log files.
:author: Jingchu Liu
:date: Nov 24 2017
"""
# ROS py
import rospy
# Message types
from std_msgs.msg import Char
from sensor_msgs.msg import CompressedImage
from autodrive_msgs.msg import CarStatus
# OpenCV related
import cv2
from cv_bridge import CvBridge
# Utils
import os
import time
import argparse
import socket
import getpass
import logging
from scipy.misc import imresize

# CvBridge
brg = CvBridge()

"""Global variable checking function."""
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
    global args
    while locked:
        print 'Global vars locked!'
        time.sleep(0.1)
    locked = True
    if new_ep:
        logging.warning(
            "[non_stop_data_capture]: "
            "detected new episode {}".format(ep_counter)
        )
        if ep_counter%video_every==0:
            if video_wrt is not None:
                video_wrt.release()
            video_wrt = cv2.VideoWriter(
                    video_file_fmt.format(ep_counter),
                    fourcc, 50.0, (320, 320))
            logging.warning(
                "[non_stop_data_capture]: "
                "created video writer to {}".format(
                    video_file_fmt.format(ep_counter)
                )
            )
            if video_info_file is not None:
                video_info_file.close()
            video_info_file = open(
                video_info_file_fmt.format(ep_counter), 'w')
        video_wrt_latest = cv2.VideoWriter(
            video_file_latest_fmt.format(ep_counter % n_latest),
            fourcc, 50.0, (320, 320))
        if dec_file is not None:
            dec_file.close()
        if args.log_decision:
            dec_file = open(
                dec_file_fmt.format(ep_counter), 'w')
        if status_file is not None:
            status_file.close()
        if args.log_status:
            status_file = open(
                status_file_fmt.format(ep_counter), 'w'
            )
    time_elapsed = now.to_nsec()
    locked = False
    new_ep = False

"""Callbacks."""
def bdview_callback(data):
    global ep_counter
    global new_ep
    now = rospy.get_rostime()
    init_global_var(now)
    seq = data.header.seq
    ts = data.header.stamp
    img = brg.compressed_imgmsg_to_cv2(data, 'bgr8')
    img = imresize(img, (320, 320))
    if not new_ep:
        if video_wrt is not None and video_info_file is not None:
            video_wrt.write(img)
            video_info_file.write("{}, {}, {}, {}\n".format(
                seq, ts.to_nsec(), now.to_nsec(), time.time()))
        if video_wrt_latest is not None:
            video_wrt_latest.write(img)

def decision_callback(data):
    global ep_counter
    now = rospy.get_rostime()
    # print "[{}: Decision @ rostime {:.3f}s] content {}".format(
    #    ep_counter, now.to_nsec()/1e9, str(data.data))
    if not new_ep and dec_file is not None:
        dec_file.write("{}, {}, {}\n".format(
            time.strftime("%Y_%b_%d_%H%M%S"), now.to_nsec(), data.data))

def status_callback(data):
    now = rospy.get_rostime()
    try:
        x, y = data.position.x, data.position.y
        if status_file is not None:
            status_file.write("{}, {}, {}, {}\n".format(
                time.strftime("%Y_%b_%d_%H%M%S"), now.to_nsec(), x, y))
    except:
        print "[{}: Status   @ rostime {:.3f}s] status data error!".format(
            ep_counter, now.to_nsec())
        pass

"""Utils."""
def to_int_or_none(name):
    try:
        return int(name[-2])
    except:
        return None


if __name__ == '__main__':
    # === Build Arguments ===
    parser = argparse.ArgumentParser()
    host = socket.gethostname()
    user = getpass.getuser()
    parser.add_argument(
        '--video_dir', type=str,
        default='./experiment/log_files_{}@{}/'.format(user, host),
        help="Directory to save recorded video files."
    )
    parser.add_argument(
        "--n_ep", type=int, default=0,
        help="Num of episode that has already been recorded. Non-positive "
             "numbers indicates infer from existing files",
    )
    parser.add_argument(
        "--video_every", type=int, default=1,
        help="Record video every this number of episodes."
    )
    parser.add_argument(
        "--n_latest", type=int, default=10,
        help="Num of latest video files to buffer on a rolling basis."
    )
    parser.add_argument(
        "--fourcc", type=str, default="X264",
        help="Four-character code of the codec used to record videos."
    )
    parser.add_argument(
        "--log_decision", action='store_true', default=False
    )
    parser.add_argument(
        "--log_status", action='store_true', default=False
    )
    args = parser.parse_args()

    # check folder and make one if non-existent
    folder = args.video_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Set file formatter strings and video recorder handles
    video_file_latest_fmt = os.sep.join([folder, 'video_ep_latest_{}.mp4'])
    video_file_fmt = os.sep.join([folder, 'video_ep_{}.mp4'])
    video_info_file_fmt = os.sep.join([folder, 'video_info_ep_{}.log'])
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    video_wrt = None
    video_info_file = None
    video_wrt_latest = None

    # Decision output log file
    dec_file_fmt = os.sep.join([folder, 'decisions_ep_{}.log'])
    dec_file = None

    # Status log file
    status_file_fmt = os.sep.join([folder, 'status_ep_{}.log'])
    status_file = None

    # episode counters
    if args.n_ep > 0:
        logging.warning(
            "[non_stop_data_capture]: "
            "using parsed n_ep {}.".format(args.n_ep)
        )
        ep_counter = args.n_ep
    else:
        files = os.listdir(folder)
        files.sort()
        files = map(lambda name: name.split('_')[-1], files)  # strip #.filetype
        files = map(lambda name: name.split('.'),
                    files)  # build list [#, filetype]
        files = map(to_int_or_none, files)
        files = [f for f in files if f is not None]
        ep_counter = max(files) + 1 if len(files) != 0 else 1
        logging.warning(
            "[non_stop_data_capture]: "
            "using n_ep inferred from existing files is {}.".format(args.n_ep)
        )

    # lock, timers
    locked = False
    time_elapsed = 1e4 * 1e9  # 1e4 seconds
    new_ep = True
    video_every = args.video_every
    n_latest = args.n_latest

    try:
        logging.warning(
            "[non_stop_data_capture]: "
            "recording episode {}...".format(ep_counter)
        )
        rospy.init_node('training_data_capture')
        rospy.Subscriber(
            "/training/image/compressed", CompressedImage, bdview_callback
        )
        if args.log_decision:
            rospy.Subscriber("/autoDrive_KeyboardMode", Char, decision_callback)
        if args.log_status:
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
    logging.warning(
        "[non_stop_data_capture]: "
        "All files closed properlly for episode {}.".format(ep_counter)
    )

