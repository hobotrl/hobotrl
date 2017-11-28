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

from playground.initialD.imitaion_learning.sl import initialD_input
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


tf.app.flags.DEFINE_string("train_dir", "./log", """save tmp model""")
tf.app.flags.DEFINE_string('checkpoint', None,
                           """Model checkpoint to load""")

FLAGS = tf.app.flags.FLAGS

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
        global_step = tf.Variable(0, trainable=False, name='global_step')
        network_train = resnet.ResNet(hp, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
        saver.restore(sess, FLAGS.checkpoint)
        graph = tf.get_default_graph()
        probs = graph.get_operation_by_name("tower_0/Softmax").outputs[0]
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
            tens_img = tf.image.resize_images(img, [224, 224])
            # img = tf.image.convert_image_dtype(img, tf.float32)
            tens_img = initialD_input.preprocess_image(tens_img)
            np_img = sess.run(tens_img)

            print "=========img shape: {}".format(img.shape)+"=========\n"


            using_learning_agent = True

            actions, np_probs = sess.run([network_train.preds, probs], feed_dict={
                network_train._images:np.array([np_img]),
                network_train.is_train:False})
            action = actions[0]

            if action != rule_action:
                print "not equal, sl: ", action, " rule: ", rule_action
                print "probs: ", np_probs
                if rule_action < 3:
                    noval_buffer.append([np.copy(np_img), rule_action])
                    noval_original_buffer.append([np.copy(img), action, rule_action, np_probs])
                    # replay_buffer.pop(0)
                    noval_scene_count += 1
            else:
                print "equal, sl&rule: ", rule_action

            # print "========\n" * 5
            next_state, reward, done, info = env.step(ACTIONS[action])
            next_img, next_rule_action = next_state
            while True:
                # print "========\n" * 5
                # print "[Delayed action] {}".format(ACTIONS[action])
                n_steps += 1
                cum_reward += reward
                next_tens_img = tf.image.resize_images(next_img, [224, 224])
                next_tens_img = initialD_input.preprocess_image(next_tens_img)
                # print "state: {}".format(next_state)
                # print "state type: {}".format(type(next_state))
                next_np_img = sess.run(next_tens_img)
                # print "np state: {}".format(np_next_state)
                next_actions, np_probs = sess.run([network_train.preds, probs], feed_dict={
                    network_train._images: np.array([next_np_img]),
                    network_train.is_train: False})
                next_action = next_actions[0]
                # r
                if next_action != next_rule_action:
                    print "not equal, sl: ", next_action, " rule: ", next_rule_action
                    print "probs: ", np_probs
                    # print "sl pred action: {}".format(next_action)
                    # print "rule action: {}".format(next_rule_action)
                    # fileter action 3 and 4
                    if next_rule_action < 3:
                        noval_buffer.append([np.copy(next_np_img), next_rule_action])
                        noval_original_buffer.append([np.copy(next_img), next_action, next_rule_action, np_probs])
                        noval_scene_count += 1
                    # replay_buffer.pop(0)
                else:
                    print "equal, sl&rule: ", next_rule_action
                    # print "sl pred and rule action: {}".format(next_rule_action)

                if done is True:
                    print "========Run Done=======\n"*5
                    break
                action = next_action  # s',a' -> s,a
                next_state, reward, done, info = env.step(ACTIONS[action])
                next_img, next_rule_action = next_state

            if noval_scene_count > 10:
                print "update_n: {}".format(n_update)
                for i, ele in enumerate(noval_original_buffer):
                    cv2.imwrite(FLAGS.train_dir+"/"+str(n_update)+"_"+
                                str(i)+"_"+str(ele[1])+"_"+str(ele[2])+"_"+str(ele[3])+".jpg", ele[0])
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
                print "conf_mat: "
                print conf_mat
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

                save_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
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

