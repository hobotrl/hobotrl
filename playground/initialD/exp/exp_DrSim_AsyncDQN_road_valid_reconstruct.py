# -*- coding: utf-8 -*-
"""Experiment script for DQN-based lane decision.

Author: Jingchu Liu
"""
# Basics
import os
import signal
import sys
import time
import traceback

import numpy as np
# Tensorflow
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from playground.initialD.exp.skip import SkippingAgent

sys.path.append('../../..')
sys.path.append('..')
# Hobotrl
import hobotrl as hrl
from hobotrl.environments import FrameStack
from hobotrl.sampling import TransitionSampler
from hobotrl.playback import BalancedMapPlayback, BigPlayback
from hobotrl.async import AsynchronousAgent
from hobotrl.utils import CappedLinear
from tensorflow.python.training.summary_io import SummaryWriterCache

# initialD
# from ros_environments.honda import DrivingSimulatorEnv
from ros_environments.clients import DrivingSimulatorEnvClient as DrivingSimulatorEnv
# Gym
from gym.spaces import Discrete, Box
import cv2


# Environment
def func_compile_reward(rewards):
    return rewards


def func_compile_obs(obss):
    obs1 = obss[-1][0]
    obs2 = obss[-2][0]
    obs3 = obss[-3][0]
    print obss[-1][1]
    # cast as uint8 is important otherwise float64
    obs = ((obs1 + obs2) / 2).astype('uint8')
    # print obs.shape
    return obs.copy()


ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
AGENT_ACTIONS = ALL_ACTIONS[:3]


# AGENT_ACTIONS = ALL_ACTIONS
def func_compile_action(action):
    ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
    return ALL_ACTIONS[action]


def func_compile_exp_agent(state, action, rewards, next_state, done):
    global cnt_skip
    global n_skip
    global momentum_opp
    global momentum_ped
    global ema_dist
    global ema_speed
    global last_road_change
    global road_invalid_at_enter

    # Compile reward
    rewards = np.mean(np.array(rewards), axis=0)
    rewards = rewards.tolist()
    rewards.append(np.logical_or(action == 1, action == 2))  # action == turn?
    print (' ' * 5 + 'R: [' + '{:4.2f} ' * len(rewards) + ']').format(*rewards),

    road_change = rewards[1] > 0.01  # road changed
    road_invalid = rewards[0] > 0.01  # any yellow or red
    if road_change and not last_road_change:
        road_invalid_at_enter = road_invalid
    last_road_change = road_change
    speed = rewards[2]
    obs_risk = rewards[5]

    ema_speed = 0.5 * ema_speed + 0.5 * speed
    ema_dist = 1.0 if rewards[6] > 2.0 else 0.9 * ema_dist
    momentum_opp = (rewards[3] < 0.5) * (momentum_opp + (1 - rewards[3]))
    momentum_opp = min(momentum_opp, 20)
    momentum_ped = (rewards[4] > 0.5) * (momentum_ped + rewards[4])
    momentum_ped = min(momentum_ped, 12)

    # road_change
    rewards[0] = -100 * (
        (road_change and ema_dist > 0.2) or (road_change and momentum_ped > 0)
    ) * (n_skip - cnt_skip)  # direct penalty
    # velocity
    rewards[2] *= 10
    rewards[2] -= 10
    # opposite
    rewards[3] = -20 * (0.9 + 0.1 * momentum_opp) * (momentum_opp > 1.0)
    # ped
    rewards[4] = -40 * (0.9 + 0.1 * momentum_ped) * (momentum_ped > 1.0)
    # obs factor
    rewards[5] *= -100.0
    # dist
    rewards[6] *= 0.0
    # steering
    rewards[-1] *= -40  # -3
    reward = np.sum(rewards) / 100.0
    print '{:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}'.format(
        road_invalid_at_enter, momentum_opp, momentum_ped, ema_dist),
    print ': {:5.2f}'.format(reward)

    if road_invalid_at_enter:
        print "[Early stopping] entered invalid road."
        road_invalid_at_enter = False
        # done = True

    if road_change and ema_dist > 0.2:
        print "[Early stopping] turned onto intersection."
        done = True

    if road_change and momentum_ped > 0:
        print "[Early stopping] ped onto intersection."
        done = True

    if obs_risk > 1.0:
        print "[Early stopping] hit obstacle."
        done = True

    return state, action, reward, next_state, done


def gen_backend_cmds():
    ws_path = '/Projects/catkin_ws/'
    initialD_path = '/Projects/hobotrl/playground/initialD/'
    backend_path = initialD_path + 'ros_environments/backend_scripts/'
    utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
    backend_cmds = [
        # Parse maps
        ['python', utils_path + 'parse_map.py',
         ws_path + 'src/Map/src/map_api/data/honda_wider.xodr',
         utils_path + 'road_segment_info.txt'],
        # Generate obs and launch file
        ['python', utils_path + 'gen_launch_dynamic_v1.py',
         utils_path + 'road_segment_info.txt', ws_path,
         utils_path + 'honda_dynamic_obs_template_tilt.launch',
         32, '--random_n_obs'],
        # start roscore
        ['roscore'],
        # start reward function script
        ['python', backend_path + 'gazebo_rl_reward.py'],
        # start road validity node script
        ['python', backend_path + 'road_validity.py',
         utils_path + 'road_segment_info.txt.signal'],
        # start car_go script
        ['python', backend_path + 'car_go.py'],
        # start simulation restarter backend
        ['python', backend_path + 'rviz_restart.py', 'honda_dynamic_obs.launch'],
        # ['python', backend_path + 'non_stop_data_capture.py', 0]
    ]
    return backend_cmds


tf.app.flags.DEFINE_string("logdir",
                           "./dqn_log",
                           """save tmp model""")
tf.app.flags.DEFINE_string("savedir",
                           "./dqn_save",
                           """records data""")
tf.app.flags.DEFINE_string("readme", "direct dqn. Use new reward function.", """readme""")
tf.app.flags.DEFINE_string("host", "10.31.40.197", """host""")
tf.app.flags.DEFINE_string("port", '10034', "Docker port")
tf.app.flags.DEFINE_string("cache_path", './dqn_ReplayBufferCache', "Replay buffer cache path")

FLAGS = tf.app.flags.FLAGS

os.mkdir(FLAGS.savedir)

env = DrivingSimulatorEnv(
    address=FLAGS.host, port=FLAGS.port,
    backend_cmds=gen_backend_cmds(),
    defs_obs=[
        ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
        ('/decision_result', 'std_msgs.msg.Int16')
    ],
    defs_reward=[
        ('/rl/current_road_validity', 'std_msgs.msg.Int16'),
        ('/rl/entering_intersection', 'std_msgs.msg.Bool'),
        ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
        ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
        ('/rl/on_biking_lane', 'std_msgs.msg.Bool'),
        ('/rl/obs_factor', 'std_msgs.msg.Float32'),
        ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
    ],
    defs_action=[('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')],
    rate_action=10.0,
    window_sizes={'obs': 3, 'reward': 3},
    buffer_sizes={'obs': 3, 'reward': 3},
    func_compile_obs=func_compile_obs,
    func_compile_reward=func_compile_reward,
    func_compile_action=func_compile_action,
    step_delay_target=0.5)
# TODO: define these Gym related params insode DrivingSimulatorEnv
env.observation_space = Box(low=0, high=255, shape=(350, 350, 3))
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
env.action_space = Discrete(len(ALL_ACTIONS))
env = FrameStack(env, 3)


# Agent
def f_net(inputs):
    inputs = inputs[0]
    inputs = inputs / 128 - 1.0
    # (640, 640, 3*n) -> ()
    conv1 = layers.conv2d(
        inputs=inputs, filters=16, kernel_size=(8, 8), strides=1,
        kernel_regularizer=l2_regularizer(scale=1e-2),
        activation=tf.nn.relu, name='conv1')
    print conv1.shape
    pool1 = layers.max_pooling2d(
        inputs=conv1, pool_size=3, strides=4, name='pool1')
    print pool1.shape
    conv2 = layers.conv2d(
        inputs=pool1, filters=16, kernel_size=(5, 5), strides=1,
        kernel_regularizer=l2_regularizer(scale=1e-2),
        activation=tf.nn.relu, name='conv2')
    print conv2.shape
    pool2 = layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=3, name='pool2')
    print pool2.shape
    conv3 = layers.conv2d(
        inputs=pool2, filters=64, kernel_size=(3, 3), strides=1,
        kernel_regularizer=l2_regularizer(scale=1e-2),
        activation=tf.nn.relu, name='conv3')
    print conv3.shape
    pool3 = layers.max_pooling2d(
        inputs=conv3, pool_size=3, strides=2, name='pool3', )
    print pool3.shape
    depth = pool3.get_shape()[1:].num_elements()
    inputs = tf.reshape(pool3, shape=[-1, depth])
    print inputs.shape
    hid1 = layers.dense(
        inputs=inputs, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='hid1')
    print hid1.shape
    hid2 = layers.dense(
        inputs=hid1, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='hid2_adv')
    print hid2.shape
    adv = layers.dense(
        inputs=hid2, units=len(AGENT_ACTIONS), activation=None,
        kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
        kernel_regularizer=l2_regularizer(scale=1e-2), name='adv')
    print adv.shape
    hid2 = layers.dense(
        inputs=hid1, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-2), name='hid2_v')
    print hid2.shape
    v = layers.dense(
        inputs=hid2, units=1, activation=None,
        kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
        kernel_regularizer=l2_regularizer(scale=1e-2), name='v')
    print v.shape
    q = tf.add(adv, v, name='q')
    print q.shape

    return {"q": q}


target_sync_rate = 1e-3
state_shape = env.observation_space.shape
graph = tf.get_default_graph()
# lr = tf.get_variable(
#     'learning_rate', [], dtype=tf.float32,
#     initializer=tf.constant_initializer(1e-3), trainable=False
# )
# lr_in = tf.placeholder(dtype=tf.float32)
# op_set_lr = tf.assign(lr, lr_in)
optimizer_td = tf.train.AdamOptimizer(learning_rate=1e-4)
global_step = tf.get_variable(
    'global_step', [], dtype=tf.int32,
    initializer=tf.constant_initializer(0), trainable=False)

# 1 sample ~= 1MB @ 6x skipping
replay_buffer = BigPlayback(
    bucket_cls=BalancedMapPlayback,
    cache_path=FLAGS.cache_path,
    capacity=300000, bucket_size=100, ratio_active=0.05, max_sample_epoch=2,
    num_actions=len(AGENT_ACTIONS), upsample_bias=(1, 1, 1, 0.1)
)

gamma = 0.9
_agent = hrl.DQN(
    f_create_q=f_net, state_shape=state_shape,
    # OneStepTD arguments
    num_actions=len(AGENT_ACTIONS), discount_factor=gamma, ddqn=True,
    # target network sync arguments
    target_sync_interval=1,
    target_sync_rate=target_sync_rate,
    # epsilon greeedy arguments
    # greedy_epsilon=0.025,
    # greedy_epsilon=0.05,
    # greedy_epsilon=0.075,
    # greedy_epsilon=0.2,  # 0.2 -> 0.15 -> 0.1
    greedy_epsilon=CappedLinear(10000, 0.15, 0.05),
    # greedy_epsilon=CappedLinear(10000, 0.1, 0.025),
    # optimizer arguments
    network_optimizer=hrl.network.LocalOptimizer(optimizer_td, 1.0),
    # sampler arguments
    sampler=TransitionSampler(replay_buffer, batch_size=8, interval=1, minimum_count=103),
    # checkpoint
    global_step=global_step
)


def log_info(update_info):
    global action_fraction
    global action_td_loss
    global agent
    global next_state
    global next_action
    global ALL_ACTIONS
    global AGENT_ACTIONS
    global n_agent_steps
    global n_env_steps
    global n_steps
    global done
    global cum_td_loss
    global cum_reward
    global n_ep
    global exploration_off
    global cnt_skip
    global n_skip
    global t_learn
    global t_infer
    global t_step
    global flag_success
    summary_proto = tf.Summary()
    # modify info dict keys
    k_del = []
    new_info = {}
    for k, v in update_info.iteritems():
        if 'FitTargetQ' in k:
            new_info[k[14:]] = v  # strip "FitTargetQ\td\"
            k_del.append(k)
    update_info.update(new_info)
    if 'td_losses' in update_info:
        prt_str = zip(
            update_info['action'], update_info['q'],
            update_info['target_q'], update_info['td_losses'],
            update_info['reward'], update_info['done'])
        for s in prt_str:
            if cnt_skip == 0:
                print ("{} " + "{:10.5f} " * 4 + "{}").format(*s)
    for tag in update_info:
        summary_proto.value.add(
            tag=tag, simple_value=np.mean(update_info[tag]))
    if 'q' in update_info and \
                    update_info['q'] is not None:
        q_vals = update_info['q']
        summary_proto.value.add(
            tag='q_vals_max',
            simple_value=np.max(q_vals))
        summary_proto.value.add(
            tag='q_vals_min',
            simple_value=np.min(q_vals))
    if cnt_skip == 0 or n_steps == 0:
        next_q_vals_nt = agent._agent.learn_q(np.asarray(next_state)[np.newaxis, :])[0]
        for i, ac in enumerate(AGENT_ACTIONS):
            summary_proto.value.add(
                tag='next_q_vals_{}'.format(ac[0]),
                simple_value=next_q_vals_nt[i])
            summary_proto.value.add(
                tag='next_q_vals_{}'.format(ac[0]),
                simple_value=next_q_vals_nt[i])
            summary_proto.value.add(
                tag='action_td_loss_{}'.format(ac[0]),
                simple_value=action_td_loss[i])
            summary_proto.value.add(
                tag='action_fraction_{}'.format(ac[0]),
                simple_value=action_fraction[i])
        p_dict = sorted(zip(
            map(lambda x: x[0], AGENT_ACTIONS), next_q_vals_nt))
        max_idx = np.argmax([v for _, v in p_dict])
        p_str = "({:.3f}) ({:3d})[Q_vals]: ".format(
            time.time(), n_steps)
        for i, (a, v) in enumerate(p_dict):
            if a == AGENT_ACTIONS[next_action][0]:
                sym = '|x|' if i == max_idx else ' x '
            else:
                sym = '| |' if i == max_idx else '   '
            p_str += '{}{:3d}: {:8.4f} '.format(sym, a, v)
        print p_str

    cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info \
                                             and update_info['td_loss'] is not None else 0
    if not done:
        cum_reward += reward

    summary_proto.value.add(tag='t_infer', simple_value=t_infer)
    summary_proto.value.add(tag='t_learn', simple_value=t_learn)
    summary_proto.value.add(tag='t_step', simple_value=t_step)

    if done:
        print ("Episode {} done in {} steps, reward is {}, "
               "average td_loss is {}. No exploration {}").format(
            n_ep, n_steps, cum_reward,
            cum_td_loss / n_steps, exploration_off)
        n_env_steps += n_steps
        summary_proto.value.add(tag='n_steps', simple_value=n_steps)
        summary_proto.value.add(tag='n_env_steps', simple_value=n_env_steps)
        summary_proto.value.add(tag='n_agent_steps', simple_value=n_agent_steps)
        summary_proto.value.add(
            tag='num_episode', simple_value=n_ep)
        summary_proto.value.add(
            tag='cum_reward', simple_value=cum_reward)
        summary_proto.value.add(
            tag='per_step_reward', simple_value=cum_reward / n_steps)
        summary_proto.value.add(
            tag='flag_success', simple_value=flag_success)

    return summary_proto


class SaveState(object):
    def __init__(self, savedir):
        self.savedir = savedir
        self.eps_dir = None
        self.file = None
        self.total_file = open(self.savedir + "/0000.txt", "w")

    def end_save(self):
        self.stat_file.close()

    def save_step(self, n_ep, n_step, state, action, vec_reward, reward,
                  done, cum_reward, flag_success, is_end=False):
        if self.file is None:
            self.eps_dir = self.savedir + "/" + str(n_ep).zfill(4)
            os.mkdir(self.eps_dir)
            self.file = open(self.eps_dir + "/0000.txt", "w")

        img = np.array(state)[:, :, 6:]
        img_path = self.eps_dir + "/" + str(n_step + 1).zfill(4) + "_" + str(action) + ".jpg"
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.file.write(str(n_steps) + ',' + str(action) + ',' + str(reward) + '\n')
        vec_reward = np.mean(np.array(vec_reward), axis=0)
        vec_reward = vec_reward.tolist()
        str_reward = ""
        for r in vec_reward:
            str_reward += str(r)
            str_reward += ","
        str_reward += "\n"
        self.file.write(str_reward)
        self.file.write("\n")
        if done:
            self.file.close()
            self.file = None
            self.stat_file.write("{}, {}, {}, {}\n".format(n_ep, cum_reward, flag_success, done))

        if is_end:
            self.end_save()


n_interactive = 0
n_skip = 6
update_rate = 6.0
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 0  # num of episode per test run (no exploration)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

agent = AsynchronousAgent(agent=_agent, method='rate', rate=update_rate)

agent = SkippingAgent(agent=agent, n_skip=6, specific_act=3)

try:
    with agent.create_session(config=config, save_dir=FLAGS.logdir, save_checkpoint_secs=3600) as sess:
        summary_writer = SummaryWriterCache.get(FLAGS.logdir)
        total_steps = 0
        saveState = SaveState(FLAGS.savedir)
        while True:
            n_ep += 1
            n_steps = 0
            state = env.reset()
            while True:
                # Do act
                action = agent.act(state)
                next_state, vec_reward, done, info = env.step(action)
                summary_writer.add_summary(info, n_steps)
                state, action, reward, next_state, done = \
                    func_compile_exp_agent(state, action, vec_reward, next_state, done)

                # save intermediate info
                saveState.save_step(n_ep, n_steps, state, action, vec_reward, reward,
                                    done, cum_reward, flag_success)
                n_steps += 1
                agent.step(state, action, reward, next_state)
                state = next_state

                summary_writer.add_summary(info, n_steps)

                if done:
                    summary = tf.Summary()
                    summary.value.add(tag="cum_reward_ep", simple_value=cum_reward)
                    summary.value.add(tag="flag_success_ep", simple_value=flag_success)
                    summary.value.add(tag="done_ep", simple_value=done)
                    summary_writer.add_summary(summary, n_ep)
                    break


except Exception as e:
    print e.message
    traceback.print_exc()
finally:
    print "=" * 30
    print "=" * 30
    print "Tidying up..."
    # kill orphaned monitor daemon process
    env.env.exit()
    replay_buffer.close()
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    print "=" * 30

