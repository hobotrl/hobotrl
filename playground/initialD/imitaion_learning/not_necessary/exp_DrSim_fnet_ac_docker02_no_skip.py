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
from playground.initialD.imitaion_learning.sl.evaluate import evaluate

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
    # action = obss[-1][1]
    # print obss[-1][1]
    # print obs1.shape
    return obs1


ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )] + [(0, )]


def func_compile_action(action):
    ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )] + [(0, )]
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
        # 5. start simulation restarter backend
        ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
        # 6. [optional] video capture
        ['python', backend_path+'non_stop_data_capture.py', 0]
    ]
    return backend_cmds


def record(summary_writer, step_n, info):
    for name in info:
        value = info[name]
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=np.mean(value))
        summary_writer.add_summary(summary, step_n)


def f_net(inputs):
    """
    action_num is set 5.
    :param inputs:
    :return:
    """
    inputs = inputs[0]
    inputs = inputs/128 - 1.0
    action_num = 5
    l2 = 1e-4
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
    inputs = tf.stop_gradient(inputs)
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
    pi = tf.stop_gradient(pi)
    q = layers.dense(inputs=hid2, units=action_num, kernel_regularizer=l2_regularizer(scale=l2), name='q')
    return {"pi": pi, "q": q}


env = DrivingSimulatorEnv(
    address="10.31.40.197", port='10004',
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
env = FrameStack(env, 3)

n_interactive = 0
n_skip = 1
n_additional_learn = 4
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)
state_shape = (350, 350, 9)
tf.app.flags.DEFINE_string("logdir",
                           "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/fnet_rename_learn_q_no_skip_tmp_debug",
                           """save tmp model""")
tf.app.flags.DEFINE_string("savedir",
                           "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q_no_skip_tmp_debug",
                           """save tmp model""")
FLAGS = tf.app.flags.FLAGS

global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
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

config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

# sv = agent.init_supervisor(
#         graph=tf.get_default_graph(), worker_index=0,
#         init_op=tf.global_variables_initializer(), save_dir=FLAGS.logdir
#     )
summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=tf.get_default_graph())

os.mkdir(FLAGS.savedir)

# restore_vars_list = ['conv1/bias', 'conv1/bias/Adadelta', 'conv1/bias/Adadelta_1', 'conv1/kernel',
#                      'conv1/kernel/Adadelta', 'conv1/kernel/Adadelta_1',
#                      'conv2/bias', 'conv2/bias/Adadelta', 'conv2/bias/Adadelta_1', 'conv2/kernel',
#                      'conv2/kernel/Adadelta', 'conv2/kernel/Adadelta_1',
#                      'conv3/bias', 'conv3/bias/Adadelta', 'conv3/bias/Adadelta_1', 'conv3/kernel',
#                      'conv3/kernel/Adadelta', 'conv3/kernel/Adadelta_1', 'global_step',
#                      'hid1/bias', 'hid1/bias/Adadelta', 'hid1/bias/Adadelta_1',
#                      'hid1/kernel', 'hid1/kernel/Adadelta', 'hid1/kernel/Adadelta_1',
#                      'hid2/bias', 'hid2/bias/Adadelta', 'hid2/bias/Adadelta_1',
#                      'hid2/kernel', 'hid2/kernel/Adadelta', 'hid2/kernel/Adadelta_1',
#                      'pi/bias', 'pi/bias/Adadelta', 'pi/bias/Adadelta_1',
#                      'pi/kernel', 'pi/kernel/Adadelta', 'pi/kernel/Adadelta_1']

restore_var_names_list = ['learn/conv1/bias:0', 'learn/conv1/kernel:0',
                     'learn/conv2/bias:0', 'learn/conv2/kernel:0',
                     'learn/conv3/bias:0', 'learn/conv3/kernel:0',
                     'learn/global_step:0',
                     'learn/hid1/bias:0', 'learn/hid1/kernel:0',
                     'learn/hid2/bias:0', 'learn/hid2/kernel:0',
                     'learn/pi/bias:0', 'learn/pi/kernel:0',]

restore_var_list = []
for var in tf.global_variables():
    print "var_name: ", var.name
    for name in restore_var_names_list:
        # print "restore_name: ", name
        if var.name == name:
            # print "restore: ", name
            restore_var_list.append(var)
            break


try:
    momentum_opp = 0.0
    momentum_ped = 0.0
    with agent.create_session(config=config, save_dir=FLAGS.logdir, restore_var_list=restore_var_list) as sess:
        while True:
            n_ep += 1
            n_steps = 0
            eps_dir = FLAGS.savedir + "/" + str(n_ep).zfill(4)
            os.mkdir(eps_dir)
            recording_filename = eps_dir + "/" + "0000.txt"
            recording_file = open(recording_filename, 'w')
            # all_scenes = []
            state = env.reset()
            print "========reset======\n" * 5
            while True:
                s1 = time.time()
                action = agent.act(state=state, evaluate=False, sess=sess)
                start = time.time()
                print "inference time: ", start - s1
                next_state, reward, done, info = env.step(action)
                end1 = time.time()
                print "env step time: ", end1 - start
                reward = func_compile_reward_agent(reward, action)
                state = np.array(state)
                img = state[:, :, :3]
                img_path = eps_dir + "/" + str(n_steps + 1).zfill(4) + \
                           "_" + str(action) + ".jpg"
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                recording_file.write(str(n_steps) + ',' + str(action) + ',' + str(reward) + '\n')
                end2 = time.time()
                print "record time: ", end2 - end1
                info = agent.step(state=state, action=action, reward=reward, next_state=next_state,
                                  episode_done=done)
                end3 = time.time()
                print "learn time: ", end3 - end2
                record(summary_writer, n_steps, info)
                end4 = time.time()
                print "summary time: ", end4 - end3
                end5 = time.time()
                print "total time: ", end5 - start

                if done is True:
                    print "========Run Done=======\n"*5
                    break
                state = next_state
                n_steps += 1

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

