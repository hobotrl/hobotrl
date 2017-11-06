# -*- coding: utf-8 -*-
"""Experiment script for DQN-based lane decision.

Author: Jingchu Liu
"""
# Basics
import os
import signal
import time
import sys
import traceback
from collections import deque
import numpy as np
# Tensorflow
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer
sys.path.append('../../..')
sys.path.append('..')
# Hobotrl
import hobotrl as hrl
from hobotrl.environments import FrameStack
from hobotrl.sampling import TransitionSampler
from hobotrl.playback import BalancedMapPlayback
# initialD
# from ros_environments.honda import DrivingSimulatorEnv
from playground.initialD.ros_environments.clients import DrivingSimulatorEnvClient as DrivingSimulatorEnv
# Gym
from gym.spaces import Discrete, Box
import cv2
import os

# Environment
def func_compile_reward(rewards):
    return rewards

def func_compile_obs(obss):
    obs1 = obss[-1][0]
    # print obss[-1][1]
    # print obs1.shape
    return obs1

ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
def func_compile_action(action):
    ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
    return ACTIONS[action]

def func_compile_reward_agent(rewards, action=0):
    global momentum_ped
    global momentum_opp
    rewards = np.mean(np.array(rewards), axis=0)
    rewards = rewards.tolist()
    rewards.append(np.logical_or(action==1, action==2))
    print (' '*10+'R: ['+'{:4.2f} '*len(rewards)+']').format(*rewards),

    # obstacle
    rewards[0] *= 0.0
    # distance to
    rewards[1] *= -10.0*(rewards[1]>2.0)
    # velocity
    rewards[2] *= 10
    # opposite
    momentum_opp = (rewards[3]<0.5)*(momentum_opp+(1-rewards[3]))
    momentum_opp = min(momentum_opp, 20)
    rewards[3] = -20*(0.9+0.1*momentum_opp)*(momentum_opp>1.0)
    # ped
    momentum_ped = (rewards[4]>0.5)*(momentum_ped+rewards[4])
    momentum_ped = min(momentum_ped, 12)
    rewards[4] = -40*(0.9+0.1*momentum_ped)*(momentum_ped>1.0)
    # obs factor
    rewards[5] *= -100.0
    # steering
    rewards[6] *= -10.0
    reward = np.sum(rewards)/100.0
    print '{:6.4f}, {:6.4f}'.format(momentum_opp, momentum_ped),
    print ': {:7.4f}'.format(reward)
    return reward


def func_compile_exp_agent(state, action, rewards, next_state, done):
    global momentum_ped
    global momentum_opp
    global ema_speed

    # Compile reward
    rewards = np.mean(np.array(rewards), axis=0)
    rewards = rewards.tolist()
    rewards.append(np.logical_or(action==1, action==2))
    print (' '*10+'R: ['+'{:4.2f} '*len(rewards)+']').format(*rewards),

    speed = rewards[2]
    ema_speed = 0.5*ema_speed + 0.5*speed
    longest_penalty = rewards[1]
    obs_risk = rewards[5]
    momentum_opp = (rewards[3]<0.5)*(momentum_opp+(1-rewards[3]))
    momentum_opp = min(momentum_opp, 20)
    # obstacle
    rewards[0] *= 0.0
    # distance to
    rewards[1] *= -10.0*(rewards[1]>2.0)
    # velocity
    rewards[2] *= 10
    # opposite
    rewards[3] = -20*(0.9+0.1*momentum_opp)*(momentum_opp>1.0)
    # ped
    momentum_ped = (rewards[4]>0.5)*(momentum_ped+rewards[4])
    momentum_ped = min(momentum_ped, 12)
    rewards[4] = -40*(0.9+0.1*momentum_ped)*(momentum_ped>1.0)
    # obs factor
    rewards[5] *= -100.0
    # steering
    rewards[6] *= -10.0
    reward = np.sum(rewards)/100.0
    print '{:6.4f}, {:6.4f}'.format(momentum_opp, momentum_ped),
    print ': {:7.4f}'.format(reward)

    # early stopping
    # if ema_speed < 0.1:
    #     if longest_penalty > 0.5:
    #         print "[Early stopping] stuck at intersection."
    #         done = True
    #     if obs_risk > 0.2:
    #         print "[Early stopping] stuck at obstacle."
    #         done = True
    #     if momentum_ped>1.0:
    #         print "[Early stopping] stuck on pedestrain."
    #         done = True

    return state, action, reward, next_state, done



def gen_backend_cmds():
    # ws_path = '/home/lewis/Projects/catkin_ws_pirate03_lowres350_dynamic/'
    ws_path = '/Projects/catkin_ws/'
    # initialD_path = '/home/lewis/Projects/hobotrl/playground/initialD/'
    initialD_path = '/Projects/hobotrl/playground/initialD/'
    backend_path = initialD_path + 'ros_environments/backend_scripts/'
    utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
    backend_cmds = [
        # 1. Parse maps
        ['python', utils_path+'parse_map.py',
         ws_path+'src/Map/src/map_api/data/honda_wider.xodr',
         utils_path+'road_segment_info.txt'],
        # 2. Generate obs and launch file
        ['python', utils_path+'gen_launch_dynamic.py',
         utils_path+'road_segment_info.txt', ws_path,
         utils_path+'honda_dynamic_obs_template.launch', 80],
        # 3. start roscore
        ['roscore'],
        # 4. start reward function script
        ['python', backend_path+'gazebo_rl_reward.py'],
        # ['python', backend_path+'rl_reward_function.py'],
        ['python', backend_path + 'car_go.py'],

        # 5. start simulation restarter backend
        ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
        # 6. [optional] video capture
        ['python', backend_path+'non_stop_data_capture.py', 0]
    ]
    return backend_cmds

env = DrivingSimulatorEnv(
    address="10.31.40.197", port='9044',
    # address='localhost', port='22224',
    backend_cmds=gen_backend_cmds(),
    defs_obs=[
        ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
        ('/decision_result', 'std_msgs.msg.Int16')
    ],
    defs_reward=[
        ('/rl/has_obstacle_nearby', 'std_msgs.msg.Bool'),
        ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
        ('/rl/car_velocity', 'std_msgs.msg.Float32'),
        ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
        ('/rl/on_pedestrian', 'std_msgs.msg.Bool'),
        ('/rl/obs_factor', 'std_msgs.msg.Float32'),
    ],
    defs_action=[('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')],
    rate_action=10.0,
    window_sizes={'obs': 2, 'reward': 3},
    buffer_sizes={'obs': 2, 'reward': 3},
    func_compile_obs=func_compile_obs,
    func_compile_reward=func_compile_reward,
    func_compile_action=func_compile_action,
    step_delay_target=0.5,
    is_dummy_action=False)
# TODO: define these Gym related params insode DrivingSimulatorEnv
env.observation_space = Box(low=0, high=255, shape=(350, 350, 3))
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
env.action_space = Discrete(len(ACTIONS))
# env = FrameStack(env, 3)

n_interactive = 0
n_skip = 3
n_additional_learn = 4
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)

tf.app.flags.DEFINE_string("save_dir", "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80_vec_rewards_all_green_docker005", """save scenes""")
FLAGS = tf.app.flags.FLAGS
try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    momentum_opp = 0.0
    momentum_ped = 0.0
    ema_speed = 10.0
    while True:
        n_ep += 1
        env.n_ep = n_ep  # TODO: do this systematically
        n_steps = 0
        state = env.reset()
        print "eps: ", n_ep
        ep_dir = FLAGS.save_dir + "/" + str(n_ep).zfill(4)
        os.makedirs(ep_dir)
        recording_file = open(ep_dir + "/" + "0000.txt", "w")

        while True:
            n_steps += 1
            action = np.random.randint(3)
            cv2.imwrite(ep_dir+"/"+str(n_steps).zfill(4)+"_"+str(action)+".jpg",
                        cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
            # Env step
            next_state, vec_reward, done, info = env.step(action)
            state, action, reward, next_state, done = func_compile_exp_agent(state, action, vec_reward, next_state, done)
            recording_file.write(str(n_steps)+","+str(action)+","+str(reward)+"\n")
            vec_reward = np.mean(np.array(vec_reward), axis=0)
            vec_reward = vec_reward.tolist()
            str_reward = ""
            for r in vec_reward:
                str_reward += str(r)
                str_reward += ","
            str_reward += "\n"
            recording_file.write(str_reward)
            # done = (reward < -0.9) or done  # heuristic early stopping
            # agent step
            state = next_state
            if done:
                break
        recording_file.close()

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

