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

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage
import sklearn.metrics

from gym.spaces import Discrete, Box
import cv2

from playground.initialD.imitaion_learning import initialD_input
import random
import hobotrl as hrl
import playground
print playground.__file__
from playground.initialD.ros_environments import DrivingSimulatorEnv


# Environment
def compile_reward(rewards):
    # print "reward 0: ", rewards[0]
    # print "reward 1: ", rewards[1]
    # print "reward 2: ", rewards[2]
    rewards = map(
        lambda reward: sum((
            -100.0 * float(reward[0]),  # obstacle 0 or -0.04
             -0.1 * float(reward[1])*(float(reward[1])>2.0),  # distance to 0.002 ~ 0.008
             0.2 * float(reward[2]),  # car_velo 0 ~ 0.08
        )),
        rewards)

    return np.mean(rewards)
    # return 1.0

def compile_obs(obss):
    obs1 = obss[-1][0]
    # rule_action = obss[-1][1]
    # obs = np.concatenate([obs1, obs2, obs3], axis=2)
    return obs1

def evaluate(y_true, preds):
    prec = sklearn.metrics.precision_score(y_true, preds, average=None)
    rec = sklearn.metrics.recall_score(y_true, preds, average=None)
    f1 = sklearn.metrics.f1_score(y_true, preds, average=None)
    conf_mat = sklearn.metrics.confusion_matrix(y_true, preds)
    return prec, rec, f1, conf_mat

env = DrivingSimulatorEnv(
    defs_obs=[('/training/image/compressed', CompressedImage)],
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
    step_delay_target=0.4
)

# def f_net(inputs):
#     l2 = 1e-3
#     state = inputs[0]
#     conv = hrl.utils.Network.conv2ds(state, shape=[(32, 4, 4), (64, 4, 4), (64, 2, 2)], out_flatten=True,
#                                      activation=tf.nn.relu,
#                                      l2=l2, var_scope="convolution")
#     q = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
#                                     l2=l2, var_scope="q")
#     pi = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
#                                      activation_out=tf.nn.softmax, l2=l2, var_scope="pi")
#     tf.stop_gradient(pi)
#     tf.stop_gradient(conv)
#     return {"q": q, "pi": pi}


def f_net(inputs, l2):
    """
    action_num is set 5.
    :param inputs:
    :return:
    """
    inputs = inputs[0]
    inputs = inputs/128 - 1.0
    action_num = 5
    # (350, 350, 3*n) -> ()
    conv1 = layers.conv2d(
        inputs=inputs, filters=16, kernel_size=(8, 8), strides=1,
        kernel_regularizer=l2_regularizer(scale=l2),
        activation=tf.nn.relu, name='conv1')
    print conv1.shape
    pool1 = layers.max_pooling2d(
        inputs=conv1, pool_size=3, strides=4, name='pool1')
    print pool1.shape
    conv2 = layers.conv2d(
        inputs=pool1, filters=16, kernel_size=(5, 5), strides=1,
        kernel_regularizer=l2_regularizer(scale=l2),
        activation=tf.nn.relu, name='conv2')
    print conv2.shape
    pool2 = layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=3, name='pool2')
    print pool2.shape
    conv3 = layers.conv2d(
         inputs=pool2, filters=64, kernel_size=(3, 3), strides=1,
         kernel_regularizer=l2_regularizer(scale=l2),
         activation=tf.nn.relu, name='conv3')
    print conv3.shape
    pool3 = layers.max_pooling2d(
        inputs=conv3, pool_size=3, strides=2, name='pool3',)
    print pool3.shape
    depth = pool3.get_shape()[1:].num_elements()
    inputs = tf.reshape(pool3, shape=[-1, depth])
    print inputs.shape
    hid1 = layers.dense(
        inputs=inputs, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=l2), name='hid1')
    print hid1.shape
    hid2 = layers.dense(
        inputs=hid1, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=l2), name='hid2')
    print hid2.shape
    pi = layers.dense(
        inputs=hid2, units=action_num, activation=tf.nn.softmax,
        kernel_regularizer=l2_regularizer(scale=l2), name='pi')
    q = layers.dense(
        inputs=hid2, units=action_num,
        kernel_initializer=l2_regularizer(scale=l2), name='q')
    return {"pi": pi, "q":q}


def preprocess_image(input_image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (input_image - imagenet_mean) / imagenet_std
    return image


def record(summary_writer, step_n, info):
    for name in info:
        value = info[name]
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=np.mean(value))
        summary_writer.add_summary(summary, step_n)

state_shape = (224, 224, 3)

global_step = tf.get_variable(
            'learn/global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

agent = hrl.ActorCritic(
            f_create_net=f_net,
            state_shape=state_shape,
            # ACUpdate arguments
            discount_factor=0.9,
            entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-2),
            target_estimator=None,
            max_advantage=100.0,
            # optimizer arguments
            network_optmizer=hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
            max_gradient=10.0,
            # sampler arguments
            sampler=None,
            batch_size=8,
            global_step=global_step,
        )

ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'd', 'a']]
n_interactive = 0
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)
reward_decay = 0.7
logdir = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/log_sl_rnd_imbalance_1"

try:
    # config = tf.ConfigProto()
    config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

    # config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = agent.init_supervisor(
        graph=tf.get_default_graph(), worker_index=0,
        init_op=tf.global_variables_initializer(), save_dir=logdir
    )
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    with sv.managed_session(config=config) as sess:
        agent.set_session(sess)
        n_steps = 0
        while True:
            n_ep += 1
            cum_reward = 0.0
            cum_td_loss = 0.0
            cum_spg_loss = 0.0
            state = env.reset()
            print "========reset======\n"*5
            img = cv2.resize(state, (224, 224))
            while True:
                action = agent.act(state=img, evaluate=False, sess=sess)
                print "action: ", action
                next_state, reward, done, info = env.step(ACTIONS[action])
                next_img = cv2.resize(next_state, (224, 224))
                n_steps += 1
                cum_reward = reward + reward_decay * cum_reward
                info = agent.step(state=img, action=action, reward=reward, next_state=next_img, episode_done=done)
                record(summary_writer, n_steps, info)
                if done is True:
                    print "========Run Done=======\n"*5
                    break
                img = next_img

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

