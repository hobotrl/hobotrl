import os
import signal
import time
import sys
import traceback
from collections import deque
sys.path.append('../../..')
sys.path.append('..')


import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.playback import MapPlayback
from playground.initialD.imitaion_learning.TmpPretrainedAgent import TmpPretrainedAgent
from hobotrl.environments.environments import FrameStack

from playground.initialD.ros_environments import DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage
import sklearn.metrics

from gym.spaces import Discrete, Box
import cv2

from playground.initialD.imitaion_learning import initialD_input
import random
import resnet

# Environment
def compile_reward(rewards):
    # rewards = map(
    #     lambda reward: sum((
    #         -100.0 * float(reward[0]),  # obstacle 0 or -0.04
    #          -1.0 * float(reward[1])*(float(reward[1])>2.0),  # distance to 0.002 ~ 0.008
    #          10.0 * float(reward[2]),  # car_velo 0 ~ 0.08
    #         -20.0 * (1 - float(reward[3])),  # opposite 0 or -0.02
    #         -70.0 * float(reward[4]),  # ped 0 ~ -0.07
    #     )),
    #     rewards)
    # return np.mean(rewards)/1000.0
    return 1.0

def compile_obs(obss):
    obs1 = obss[-1][0]
    rule_action = obss[-1][1]
    # obs = np.concatenate([obs1, obs2, obs3], axis=2)
    return obs1, rule_action

def evaluate(y_true, preds):
    prec = sklearn.metrics.precision_score(y_true, preds, average=None)
    rec = sklearn.metrics.recall_score(y_true, preds, average=None)
    f1 = sklearn.metrics.f1_score(y_true, preds, average=None)
    conf_mat = sklearn.metrics.confusion_matrix(y_true, preds)
    return prec, rec, f1, conf_mat
    # print "val_prec: {}".format(prec)
    # print "val_rec: {}".format(rec)
    # print "val_f1: {}".format(f1)
    # print "val_conf_mat: {}".format(conf_mat)


def f_net(inputs):
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    # saver.restore(sess, checkpoint)
    state = inputs[0]
    res = resnet.ResNet(hp, global_step, name="train")
    pi = res.build_origin_tower(state)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=12000)
    checkpoint = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/exp/rename_net/resnet_log3_2"
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
    print "pi type: ", type(pi)
    pi = tf.nn.softmax(pi)
    # q = res.build_new_tower(state)
    # print "q type: ", type(q)
    # return {"q":q, "pi": pi}
    return pi

tf.app.flags.DEFINE_string("train_dir", "./log_test_fnet", """save tmp model""")


FLAGS = tf.app.flags.FLAGS


if not os.path.exists(FLAGS.train_dir):
    os.mkdir(FLAGS.train_dir)
else:
    sys.exit(1)

# What is the result's name?? Need check
env = DrivingSimulatorEnv(
    defs_obs=[('/training/image/compressed', CompressedImage),
              ('/decision_result', Int16)],
    func_compile_obs=compile_obs,
    defs_reward=[
        ('/rl/has_obstacle_nearby', Bool),
        ('/rl/distance_to_longestpath', Float32),
        ('/rl/car_velocity', Float32),
        ('/rl/last_on_opposite_path', Int16),
        ('/rl/on_pedestrian', Bool)],
    func_compile_reward=compile_reward,
    defs_action=[('/autoDrive_KeyboardMode', Char)],
    rate_action=10.0,
    window_sizes={'obs': 2, 'reward': 3},
    buffer_sizes={'obs': 2, 'reward': 3},
    step_delay_target=0.5
)
env.observation_space = Box(low=0, high=255, shape=(640, 640, 3))
env.action_space = Discrete(3)
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
# env = FrameStack(env, 1)
ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'd', 'a']]


state_shape = env.observation_space.shape
# graph = tf.get_default_graph()

n_interactive = 0
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)
n_update = 0

# filename = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/train.tfrecords"
# replay_buffer = initialD_input.init_replay_buffer(filename, replay_size=10000, batch_size=200)
all_scenes = []
noval_buffer = []
noval_original_buffer = []
noval_scene_count = 0
batch_size = 256


try:
    # config = tf.ConfigProto()
    config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

    with tf.Session(config=config) as sess:
        hp = resnet.HParams(batch_size=batch_size,
                            num_gpus=1,
                            num_classes=3,
                            weight_decay=0.001,
                            momentum=0.9,
                            finetune=True)
        global_step = tf.Variable(0, trainable=False, name='learn/global_step')
        network_train = resnet.ResNet(hp, global_step, name="train")
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        pi = network_train.build_origin_tower(images)
        # network_train.build_train_op()
        init = tf.global_variables_initializer()
        sess.run(init)
        # graph = tf.get_default_graph()
        # probs = graph.get_operation_by_name("learn/tower_0/Softmax").outputs[0]
        # graph = tf.get_default_graph()

        print "agent initialization done"
        # print "========\n"*5
        # lr = graph.get_operation_by_name('lr').outputs[0]
        while True:
            n_ep += 1
            cum_reward = 0.0
            n_steps = 0
            cum_td_loss = 0.0
            img, rule_action = env.reset()
            # print "state shape: {}".format(state.shape)
            # print "state type: {}".format(type(state))
            # resize maybe different from tf.resize
            # tensor_state = tf.convert_to_tensor(state)
            # img = np.array([img])
            # tens_img = tf.image.resize_images(img, [224, 224])
            # img = tf.image.convert_image_dtype(img, tf.float32)
            # tens_img = initialD_input.preprocess_image(tens_img)
            # np_img = sess.run(tens_img)
            img = cv2.resize(img, (224, 224))
            np_img = initialD_input.preprocess_image(img)
            print "=========img shape: {}".format(img.shape)+"=========\n"

            # actions, np_probs = sess.run([network_train.preds, probs], feed_dict={
            #     network_train._images:np.array([np_img]),
            #     network_train.is_train:False})
            # action = actions[0]

            probs = sess.run(pi, feed_dict={images:np.array([np_img])})
            action = np.argmax(probs)
            all_scenes.append([np.copy(img), action, probs])
            next_state, reward, done, info = env.step(ACTIONS[action])
            next_img, next_rule_action = next_state
            while True:
                n_steps += 1
                cum_reward += reward
                # next_tens_img = tf.image.resize_images(next_img, [224, 224])
                # next_tens_img = initialD_input.preprocess_image(next_tens_img)
                # next_np_img = sess.run(next_tens_img)
                next_img = cv2.resize(next_img, (224, 224))
                next_np_img = initialD_input.preprocess_image(next_img)
                # next_actions, np_probs = sess.run([network_train.preds, probs], feed_dict={
                #     network_train._images: np.array([next_np_img]),
                #     network_train.is_train: False})
                # next_action = next_actions[0]
                probs = sess.run(pi, feed_dict={images:np.array([next_np_img])})
                next_action = np.argmax(probs)
                print next_action
                all_scenes.append([np.copy(next_img), next_action, probs])
                if done is True:
                    print "========Run Done=======\n"*5
                    break
                action = next_action  # s',a' -> s,a
                next_state, reward, done, info = env.step(ACTIONS[action])
                # next_state, reward, done, info = env.step(ACTIONS[action])
                # next_state, reward, done, info = env.step(ACTIONS[action])
                next_img, next_rule_action = next_state

            for i, ele in enumerate(all_scenes):
                cv2.imwrite(FLAGS.train_dir + "/" + str(n_ep) + "_" +
                            str(i) + "_" + str(ele[1]) + "_" + str(np.around(ele[2], 2)) + ".jpg", ele[0])


except Exception as e:
    print e.message
    traceback.print_exc()
finally:
    print "="*30
    print "="*30
    print "Tidying up..."
    # kill orphaned monitor daemon process
    env.exit()
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    print "="*30

