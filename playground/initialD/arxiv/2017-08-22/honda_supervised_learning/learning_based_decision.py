import rospy
import numpy as np
from scipy.misc import imresize
import tensorflow as tf
from std_msgs.msg import Char
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from pyautogui import typewrite
import time
import matplotlib.pyplot as plt
# CvBridge
brg = CvBridge()
ckpg_file = "overfit_640_D500_REST.ckpt-130000"
saver = tf.train.import_meta_graph('./'+ckpg_file+'.meta',
                                   clear_devices=True)
sess = tf.Session(
#    config=tf.ConfigProto(device_count={"GPU": 0, "CPU": 1})
)
saver.restore(sess, './'+ckpg_file)
graph = tf.get_default_graph()
frame = graph.get_operation_by_name('frame').outputs[0]
logits = graph.get_operation_by_name('logits').outputs[0]
keys = {0: 's', 1: 'd', 2: 'a', 3:'?'}

def bdview_callback(data, pub):
    now = rospy.get_rostime()
    seq = data.header.seq
    ts = data.header.stamp
    img = brg.imgmsg_to_cv2(data, 'rgb8')
    img = imresize(img, (640, 640))
    print "[Video    @ rostime {:.3f}s] seq # {}, delay {}ms".format(
        now.to_nsec()/1e9, seq, (now.to_nsec()-ts.to_nsec())/1e6
    )
    logits_val =sess.run(
        logits, feed_dict={frame: img[np.newaxis, :]}
    )
    now = rospy.get_rostime()
    delay = (now.to_nsec()-ts.to_nsec())/1e6
    print "[Decision @ rostime {:.3f}s] seq # {}, delay {}ms".format(
        now.to_nsec()/1e9, seq, delay
    ),
    print (
        "["+"{:.4f}, "*np.shape(logits_val)[1]+"]"
    ).format(*logits_val.tolist()[0]),
    print np.argmax(logits_val, axis=1).flatten()
    # if np.argmax(logits_val, axis=1).flatten()[0]>0:
    #    plt.imshow(img)
    #    plt.show()
    mode = keys[np.argmax(logits_val, axis=1)[0]]
    pub.publish(Char(ord(mode)))

def decide():
    rospy.init_node('learning_based_decision')
    pub = rospy.Publisher('/autoDrive_KeyboardMode', Char, queue_size=10)
    rospy.Subscriber("/training/image", Image, lambda data:
                     bdview_callback(data, pub))
    rospy.spin()

if __name__ == '__main__':
    try:
        decide()
    except rospy.ROSInterruptException:
        pass
