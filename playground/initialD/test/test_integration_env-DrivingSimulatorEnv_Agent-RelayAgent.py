import os
import signal
import time
import sys
import traceback
sys.path.append('../../..')
sys.path.append('..')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.playback import MapPlayback
from hobotrl.algorithms.dqn import DQN

from relay_agent import RelayAgent
from ros_environments import DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import Image

# Environment
def compile_reward(rewards):
    rewards = rewards[0]
    reward = -100.0 * float(rewards[0]) + \
              -10.0 * float(rewards[1]) + \
               10.0 * float(rewards[2]) + \
             -100.0 * (1 - float(rewards[3]))
    return reward

def compile_obs(obss):
    obs = obss[0][0]
    return obs

env = DrivingSimulatorEnv(
    defs_obs=[('/training/image', Image)],
    func_compile_obs=compile_obs,
    defs_reward=[
        ('/rl/has_obstacle_nearby', Bool),
        ('/rl/distance_to_longestpath', Float32),
        ('/rl/car_velocity', Float32),
        ('/rl/last_on_opposite_path', Int16)],
    func_compile_reward=compile_reward,
    defs_action=[('/autoDrive_KeyboardMode', Char)],
    rate_action=1e-10,
    window_sizes={'obs': 1, 'reward': 1},
    buffer_sizes={'obs': 5, 'reward': 5},
    step_delay_target=1.0,
    is_dummy_action=True
)
ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'd', 'a', '!']]

# Agent
agent = RelayAgent(queue_len=5)

n_interactive = 0

try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    agent.set_session(sess)
    while True:
        cum_reward = 0.0
        n_steps = 0
        cum_td_loss = 0.0
        state, action = env.reset(), np.random.randint(0, len(ACTIONS))
        next_state, reward, done, info = env.step(ACTIONS[action])
        while True:
            n_steps += 1
            cum_reward += reward
            next_action, update_info = agent.step(
                sess=sess,
                state=map(lambda x: (x-2)/5.0, state),  # scale state to [-1, 1]
                action=action,
                reward=float(reward>1.0),  # reward clipping
                next_state=map(lambda x: (x-2)/5.0, next_state), # scle state
                episode_done=done,
            )
            cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info is not None else 0
            # print update_info
            if done is True:
                print "Episode done in {} steps, reward is {}, average td_loss is {}".format(
                    n_steps, cum_reward, cum_td_loss/n_steps
                )
                n_steps = 0
                cum_reward = 0.0
                break
            state, action = next_state, next_action
            next_state, reward, done, info = env.step(ACTIONS[action])
#except rospy.ROSInterruptException:
except Exception as e:
    print e.message
finally:
    print "Tidying up..."
    sess.close()
    # kill orphaned monitor daemon process
    os.killpg(os.getpgid(os.getpid()), 9)
