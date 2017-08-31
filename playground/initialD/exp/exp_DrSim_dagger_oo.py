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
from playground.resnet import resnet

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

filename = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/train.tfrecords"
replay_buffer = initialD_input.init_replay_buffer(filename, replay_size=10000, batch_size=200)
noval_buffer = []
noval_scene_count = 0
batch_size = 256


try:
    # config = tf.ConfigProto()
    config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
        allow_soft_placement=True,
        log_device_placement=False)

    with tf.Session(config=config) as sess:
        train_dir = "./tmp"
        hp = resnet.HParams(batch_size=batch_size,
                            num_gpus=1,
                            num_classes=3,
                            weight_decay=0.001,
                            momentum=0.9,
                            finetune=True)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        network_train = resnet.ResNet(hp, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
        checkpoint = "/home/pirate03/PycharmProjects/resnet-18-tensorflow/log3_tmp/model.ckpt-10000"
        saver.restore(sess, checkpoint)

        # graph = tf.get_default_graph()
        # tensor_imgs = graph.get_tensor_by_name('images:0')
        # tensor_acts = graph.get_tensor_by_name('labels:0')
        # preds = graph.get_tensor_by_name('tower_0/ToInt32:0')
        # train_op = graph.get_operation_by_name("group_deps")
        # is_train = graph.get_operation_by_name('is_train').outputs[0]
        # lr = graph.get_operation_by_name('lr').outputs[0]
        # print "========\n"*5
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
            img = tf.image.resize_images(img, [224, 224])
            # img = tf.image.convert_image_dtype(img, tf.float32)
            img = initialD_input.preprocess_image(img)
            img = sess.run(img)

            print "=========img shape: {}".format(img.shape)+"=========\n"


            using_learning_agent = True

            action = sess.run(network_train.preds, feed_dict={
                network_train._images:np.array([img]),
                network_train.is_train:False})[0]

            if action != rule_action:
                print "not identical "
                print "sl pred action: {}".format(action)
                print "rule action: {}".format(rule_action)

                if rule_action != 3:
                    noval_buffer.append([np.copy(img), rule_action])
                    # replay_buffer.pop(0)
                    noval_scene_count += 1
            else:
                print "identical"
                print "sl pred and rule action: {}".format(rule_action)

            # print "========\n" * 5
            next_state, reward, done, info = env.step(ACTIONS[action])
            next_img, next_rule_action = next_state
            while True:
                # print "========\n" * 5
                # print "[Delayed action] {}".format(ACTIONS[action])
                n_steps += 1
                cum_reward += reward
                next_img = tf.image.resize_images(next_img, [224, 224])
                next_img = initialD_input.preprocess_image(next_img)
                # print "state: {}".format(next_state)
                # print "state type: {}".format(type(next_state))
                next_img = sess.run(next_img)
                # print "np state: {}".format(np_next_state)
                next_action = sess.run(network_train.preds, feed_dict={
                    network_train._images: np.array([next_img]),
                    network_train.is_train: False})[0]
                # r
                if next_action != next_rule_action:
                    print "not identical"
                    print "sl pred action: {}".format(next_action)
                    print "rule action: {}".format(next_rule_action)
                    # fileter action 3
                    if next_rule_action != 3:
                        noval_buffer.append([np.copy(next_img), next_rule_action])
                        noval_scene_count += 1
                    # replay_buffer.pop(0)
                else:
                    print "identical"
                    print "sl pred and rule action: {}".format(next_rule_action)

                if done is True:
                    print "========Run Done=======\n"*5
                    break
                img, action = next_img, next_action  # s',a' -> s,a
                next_state, reward, done, info = env.step(ACTIONS[action])
                next_img, next_rule_action = next_state

            if noval_scene_count > 10:
                print "update_n: {}".format(n_update)
                print "========Trying to learn======\n"*5
                replay_size = len(replay_buffer)
                batch_size = 256
                # num_ = replay_size * 10 / batch_size
                val_replay_num = 500
                val_noval_num = 12
                val_set = [random.choice(replay_buffer) for i in range(val_replay_num)]
                val_set.extend([random.choice(noval_buffer) for i in range(val_noval_num)])
                val_imgs = np.array([val_set[i][0] for i in range(val_replay_num+val_noval_num)])
                val_acts = np.array([val_set[i][1] for i in range(val_replay_num+val_noval_num)])
                # batch = replay_buffer[np.random.randint(replay_size, size=batch_size)]
                y_preds = sess.run(network_train.preds,
                                   feed_dict={network_train._images: val_imgs,
                                            network_train.is_train: False})
                # print "y_true: ", batch_acts
                # print "y_preds: ", y_preds
                prec, rec, f1, conf_mat = evaluate(val_acts, y_preds)
                print "before learning:  ", "prec: ", prec, "rec: ", rec
                print "conf_mat: ", conf_mat
                # y_true = np.array([y[1] for y in replay_buffer])

                train_num = 10
                noval_cent = 5
                for i in range(train_num):
                    print "learning timestep.... {}".format(i)
                    print "noval buffer size: ", len(noval_buffer)
                    batch = [random.choice(replay_buffer) for j in range(batch_size-noval_cent)]
                    batch.extend([random.choice(noval_buffer) for j in range(noval_cent)])
                    batch_imgs = np.array([batch[j][0] for j in range(batch_size)])
                    batch_acts = np.array([batch[j][1] for j in range(batch_size)])
                    sess.run(network_train.train_op, feed_dict={network_train._images: batch_imgs,
                                                    network_train._labels: batch_acts,
                                                    network_train.is_train: True,
                                                    network_train.lr: 0.001})
                n_update += 1

                # batch = replay_buffer[np.random.randint(replay_size, size=batch_size)]
                y_preds = sess.run(network_train.preds,
                                   feed_dict={network_train._images: val_imgs,
                                              network_train.is_train: False})
                # print "y_true: ", batch_acts
                # print "y_preds: ", y_preds
                prec, rec, f1, conf_mat = evaluate(val_acts, y_preds)
                print "prec: ", prec
                print "rec: ", rec
                print "conf_mat: ", conf_mat

                save_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, save_path, global_step= n_update * train_num)
                print "=======Learning Done======\n"*5
                noval_scene_count = 0


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

