import os
import signal
import time
import sys
import traceback
from collections import deque
sys.path.append('../../..')
sys.path.append('..')

from playground.initialD.imitaion_learning import resnet

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.playback import MapPlayback
from dqn import DQNSticky
from hobotrl.environments.environments import FrameStack

from ros_environments import DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage

from gym.spaces import Discrete, Box
import cv2

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
    obs1 = obss[-1][0]
    # obs = np.concatenate([obs1, obs2, obs3], axis=2)
    return obs1

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
    step_delay_target=0.3
)
env.observation_space = Box(low=0, high=255, shape=(640, 640, 3))
env.action_space = Discrete(3)
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
# env = FrameStack(env, 1)
ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'a', 'd']]


state_shape = env.observation_space.shape
# graph = tf.get_default_graph()

n_interactive = 0
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)

def preprocess_image(input_image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (input_image - imagenet_mean) / imagenet_std

    return image


try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # sv = tf.train.Supervisor(
    #     graph=tf.get_default_graph(),
    #     is_chief=True,
    #     init_op=tf.global_variables_initializer(),
    #     logdir='./experiment',
    #     save_summaries_secs=20,
    #     save_model_secs=1800)

    # recover


    # inputs = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 3])
    # resized_inputs = tf.image.resize_images(inputs, [224, 224])
    # preds = f_net_policy(resized_inputs, 3)


    # with sv.managed_session(config=config) as sess:
    with tf.Session() as sess:
        # agent.set_session(sess)
        checkpoint = "/home/pirate03/PycharmProjects/resnet-18-tensorflow/log2_15/model.ckpt-9999"
        saver = tf.train.import_meta_graph('/home/pirate03/PycharmProjects/resnet-18-tensorflow/log2_15/model.ckpt-9999.meta',
                                           clear_devices=True)
        saver.restore(sess, checkpoint)
        graph = tf.get_default_graph()

        inputs = graph.get_tensor_by_name('train_image/shuffle_batch:0')
        preds = graph.get_tensor_by_name('tower_0/ToInt32:0')
        is_train = graph.get_operation_by_name('is_train').outputs[0]
        # lr = graph.get_operation_by_name('lr').outputs[0]
        while True:
            n_ep += 1
            cum_reward = 0.0
            n_steps = 0
            cum_td_loss = 0.0
            exploration_off = (n_ep%n_test==0)
            state = env.reset()
            print "state shape: ", state.shape
            state = cv2.resize(state, (224, 224))
            state = np.array(state, dtype=np.float32)
            state /= 255.0
            state = preprocess_image(state)
            action = sess.run(preds, feed_dict={
                inputs: np.repeat(
                    state[np.newaxis, :, :, :],
                    axis=0,
                    repeats=256),
                is_train: False,
            })[0]
            # action = agent.act(state, exploration_off=exploration_off)
            next_state, reward, done, info = env.step(ACTIONS[action])
            queue = deque([(state, 0)]*1)
            queue.append((state, action))
            state, action = queue.popleft()
            while True:
                print "[Delayed action] {}".format(ACTIONS[action])
                n_steps += 1
                cum_reward += reward
                # _, update_info = agent.step(
                #     sess=sess, state=state, action=action,
                #     reward=reward, next_state=next_state,
                #     episode_done=done,
                #     learning_off=exploration_off,
                #     exploration_off=exploration_off)

                next_state = cv2.resize(next_state, (224, 224))
                next_state = np.array(next_state, dtype=np.float32)
                next_state /= 255.0
                next_state = preprocess_image(next_state)
                next_action = sess.run(preds, feed_dict={
                    inputs: np.repeat(
                        next_state[np.newaxis, :, :, :],
                        axis=0,
                        repeats=256),
                    is_train: False,
                })[0]

                if done is True:
                    break
                state, action = next_state, next_action  # s',a' -> s,a
                next_state, reward, done, info = env.step(ACTIONS[action])
                queue.append((state, action))
                state, action = queue.popleft()
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

