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
import hobotrl as hrl

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

def f_net(inputs):
    l2 = 1e-4
    state = inputs[0]
    conv = hrl.utils.Network.conv2ds(state, shape=[(32, 4, 4), (64, 4, 4), (64, 2, 2)], out_flatten=True,
                                     activation=tf.nn.relu,
                                     l2=l2, var_scope="convolution")
    q = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
                                    l2=l2, var_scope="q")
    pi = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
                                     activation_out=tf.nn.softmax, l2=l2, var_scope="pi")
    return {"q": q, "pi": pi}


tf.app.flags.DEFINE_string("train_dir", "./DrSim_fnet_sl", """save tmp model""")
# tf.app.flags.DEFINE_string('checkpoint',
#     "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/DrSim_fnet_sl/model.ckpt-0",
#                            """Model checkpoint to load""")

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
    step_delay_target=0.4
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

imbalance_factor = np.array([1.0, 1.0, 1.0])

global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
input_state = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
pi = f_net([input_state])['pi']


try:
    # config = tf.ConfigProto()
    config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

    sv = tf.train.Supervisor(graph=tf.get_default_graph(),
                             global_step=global_step,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             logdir=FLAGS.train_dir,
                             save_summaries_secs=0)

    with sv.managed_session() as sess:
        print "agent initialization done"
        while True:
            n_ep += 1
            cum_reward = 0.0
            n_steps = 0
            cum_td_loss = 0.0
            img, rule_action = env.reset()
            np_img = cv2.resize(img, (224, 224))
            print "=========img shape: {}".format(img.shape)+"=========\n"
            np_probs = sess.run(pi, feed_dict={input_state:np.array([np_img])})
            ib_np_probs = np_probs * imbalance_factor
            action = np.argmax(ib_np_probs)
            print "actions: ", action
            all_scenes.append([np.copy(img), action, np_probs, ib_np_probs])
            next_state, reward, done, info = env.step(ACTIONS[action])
            next_img, next_rule_action = next_state
            while True:
                n_steps += 1
                cum_reward += reward
                next_np_img = cv2.resize(next_img, (224, 224))
                np_probs = sess.run(pi, feed_dict={input_state: np.array([next_np_img])})
                ib_np_probs = np_probs * imbalance_factor
                next_action = np.argmax(ib_np_probs)
                print next_action
                all_scenes.append([np.copy(next_img), next_action, np_probs, ib_np_probs])
                if done is True:
                    print "========Run Done=======\n"*5
                    break
                action = next_action  # s',a' -> s,a
                print "action: ", action
                next_state, reward, done, info = env.step(ACTIONS[action])
                next_img, next_rule_action = next_state


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

