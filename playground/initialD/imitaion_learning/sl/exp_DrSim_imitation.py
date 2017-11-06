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

from gym.spaces import Discrete, Box
import cv2

from playground.initialD.imitaion_learning.sl import initialD_input


# Environment
def compile_reward(rewards):
    rewards = map(
        lambda reward: sum((
            -100.0 * float(reward[0]),  # obstacle 0 or -0.04
             -1.0 * float(reward[1])*(float(reward[1])>2.0),  # distance to 0.002 ~ 0.008
             10.0 * float(reward[2]),  # car_velo 0 ~ 0.08
            -20.0 * (1 - float(reward[3])),  # opposite 0 or -0.02
            -70.0 * float(reward[4]),  # ped 0 ~ -0.07
        )),
        rewards)
    return np.mean(rewards)/1000.0

def compile_obs(obss):
    obs1 = obss[0][0]
    rule_action = obss[-1][0]
    # obs = np.concatenate([obs1, obs2, obs3], axis=2)
    return obs1, rule_action


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
    step_delay_target=0.7
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


filename = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/train.tfrecords"
print "~~~~~~~\n"*5
replay_buffer = initialD_input.init_replay_buffer(filename)
print "~~~~~~~\n"*5
print "replay_buffer initialization done"
noval_scene_count = 0

try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:

        checkpoint_path = "/home/pirate03/PycharmProjects/resnet-18-tensorflow/log2_15/model.ckpt-9999"
        graph = tf.get_default_graph()
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
        input_name = 'train_image/shuffle_batch:0'
        output_name = 'tower_0/ToInt32:0'
        print "agent initialization"
        train_dir = "./tmp"
        pretrained_agent = TmpPretrainedAgent(checkpoint_path, train_dir, input_name, output_name,
                                              sess, graph, global_step)
        print "agent initialization done"
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

            img = tf.image.resize_images(img, [224, 224])
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = initialD_input.preprocess_image(img)
            # print "state: {}".format(state)
            # print "state type: {}".format(type(state))
            img = sess.run(img)
            # print "np state: {}".format(np_state)
            # state = cv2.resize(state, (224, 224))
            # state = np.array(state, dtype=np.float32)
            # state /= 255.0
            # state = preprocess_image(state)

            using_learning_agent = True
            action = pretrained_agent.act(img)
            if action != rule_action:
                print "not identical "
                print "sl pred action: {}".format(action)
                print "rule action: {}".format(rule_action)
                if rule_action != 3:
                    replay_buffer.append([np.copy(img), action])
                    replay_buffer.pop(0)
                    noval_scene_count += 1
            else:
                print "identical"
            # default using learning agent so three is no need to step('1')
            # env.step(ACTIONS[5])

            # if is_common_scene(img):
            #     env.step(ACTIONS[5])
            #     using_learning_agent = True
            #     action = pretrained_agent.act(img)
            # else:
            #     # not sure '0' or '1'
            #     # HOW TO USE SUBSCRIBER
            #     env.step(ACTIONS[4])
            #     using_learning_agent = False
            #     action = rule_action
            #     noval_scene_count += 1
            #     replay_buffer.append([np.copy(img), action])
                # replay_buffer.pop(0)

            print "action: {}".format(action)
            next_img, next_rule_action, reward, done, info = env.step(ACTIONS[action])

            while True:
                print "[Delayed action] {}".format(ACTIONS[action])
                n_steps += 1
                cum_reward += reward
                next_img = tf.image.resize_images(next_img, [224, 224])
                next_img = tf.image.convert_image_dtype(next_img, tf.float32)
                next_img = initialD_input.preprocess_image(next_img)
                # print "state: {}".format(next_state)
                # print "state type: {}".format(type(next_state))
                next_img = sess.run(next_img)
                # print "np state: {}".format(np_next_state)
                next_action = pretrained_agent.act(next_img)
                # r
                if next_action != next_rule_action:
                    print "not identical"
                    # fileter action 3
                    if next_rule_action != 3:
                        replay_buffer.append([np.copy(next_img), next_action])
                        replay_buffer.pop(0)
                        noval_scene_count += 1
                    # replay_buffer.pop(0)
                else:
                    print "identical"

                print "action: {}".format(next_action)

                if done is True:
                    break
                img, action = next_img, next_action  # s',a' -> s,a
                next_img, next_rule_action, reward, done, info = env.step(ACTIONS[action])

            if noval_scene_count > 20:
                print "Trying to learn"
                pretrained_agent.learn(replay_buffer)
                print "Learning Done"
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

