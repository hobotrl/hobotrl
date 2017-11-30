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

sys.path.append('../../..')
sys.path.append('..')

from playground.initialD.exp.skipping_masking import SkippingAgent

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


class FuncReward(object):
    def __init__(self):
        self.mom_opp = 0.0
        self.mom_biking = 0.0
        self.ema_speed = 10.0
        self.ema_dist = 0.0
        self.last_road_change = False
        self.road_invalid_at_enter = False

    def reset(self):
        self.mom_opp = 0.0
        self.mom_biking = 0.0
        self.ema_speed = 10.0
        self.ema_dist = 0.0
        self.last_road_change = False
        self.road_invalid_at_enter = False

    def func(self, action, rewards, done, n_skip=1, cnt_skip=0):
        rewards = np.mean(np.array(rewards), axis=0)
        rewards = rewards.tolist()
        rewards.append(np.logical_or(action == 1, action == 2))  # action == turn?
        print (' ' * 5 + 'R: [' + '{:4.2f} ' * len(rewards) + ']').format(*rewards),

        speed = rewards[0]
        dist = rewards[1]
        obs_risk = rewards[2]
        road_invalid = rewards[3] > 0.01  # any yellow or red
        road_change = rewards[4] > 0.01  # entering intersection
        opp = rewards[5]
        biking = rewards[6]
        # inner = rewards[7]
        # outter = rewards[8]
        steer = rewards[-1]

        self.ema_speed = 0.5 * self.ema_speed + 0.5 * speed
        if road_change and not self.last_road_change:
            self.road_invalid_at_enter = road_invalid
        self.last_road_change = road_change
        self.ema_dist = 1.0 if dist > 2.0 else 0.9 * self.ema_dist
        self.mom_opp = (opp < 0.5) * (self.mom_opp + 1)
        self.mom_opp = min(self.mom_opp, 20)
        self.mom_biking = (biking > 0.5) * (self.mom_biking + 1)
        self.mom_biking = min(self.mom_biking, 12)
        print '{:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}'.format(
            self.road_invalid_at_enter, self.mom_opp, self.mom_biking, self.ema_dist),

        reward = []
        # velocity
        reward.append(speed * 10 - 10)
        # banned road change
        reward.append(-100 * (
            (road_change and self.ema_dist > 0.2) or (road_change and self.mom_biking > 0)
        ) * (n_skip - cnt_skip))
        # obs factor
        reward.append(-100.0 * obs_risk)
        # opposite
        reward.append(-20 * (0.9 + 0.1 * self.mom_opp) * (self.mom_opp > 1.0))
        # ped
        reward.append(-40 * (0.9 + 0.1 * self.mom_biking) * (self.mom_biking > 1.0))
        # steering
        reward.append(steer * -40.0)
        reward = np.sum(reward) / 100.0
        print ': {:5.2f}'.format(reward)

        if self.road_invalid_at_enter:
            print "[Early stopping] entered invalid road."
            self.road_invalid_at_enter = False
            # done = True
        if road_change and self.ema_dist > 0.2:
            print "[Early stopping] turned  intersection."
            done = True
        if road_change and self.mom_biking > 0:
            print "[Early stopping] ped onto intersection."
            done = True
        if obs_risk > 1.0:
            print "[Early stopping] hit obstacle."
            done = True

        if done:
            self.reset()

        return reward, done



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
        ('/decision_result', 'std_msgs.msg.Int16'),
        ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
    ],
    defs_reward=[
        ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
        ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
        ('/rl/obs_factor', 'std_msgs.msg.Float32'),
        ('/rl/current_road_validity', 'std_msgs.msg.Int16'),
        ('/rl/entering_intersection', 'std_msgs.msg.Bool'),
        ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
        ('/rl/on_biking_lane', 'std_msgs.msg.Bool'),
        ('/rl/on_innerest_lane', 'std_msgs.msg.Bool'),
        ('/rl/on_outterest_lane', 'std_msgs.msg.Bool')
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


class Logger(object):
    def __init__(self):
        self.cum_td_loss = 0.0
        self.cum_reward = 0.0

    def reset(self):
        self.cum_td_loss = 0.0
        self.cum_reward = 0.0

    def log(self, update_info, state, action, n_ep, total_steps, n_steps):
        if update_info == {}:
            return
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
        # q_vals_nt = agent._agent.learn_q(np.asarray(state)[np.newaxis, :])[0]
        # for i, ac in enumerate(AGENT_ACTIONS):
        #     summary_proto.value.add(
        #         tag='q_vals_{}'.format(ac[0]),
        #         simple_value=q_vals_nt[i])
        #     summary_proto.value.add(
        #         tag='q_vals_{}'.format(ac[0]),
        #         simple_value=q_vals_nt[i])
        #     p_dict = sorted(zip(
        #         map(lambda x: x[0], AGENT_ACTIONS), q_vals_nt))
        #     max_idx = np.argmax([v for _, v in p_dict])
        #     p_str = "({:.3f}) ({:3d})[Q_vals]: ".format(
        #         time.time(), n_steps)
        #     for i, (a, v) in enumerate(p_dict):
        #         if a == AGENT_ACTIONS[action][0]:
        #             sym = '|x|' if i == max_idx else ' x '
        #         else:
        #             sym = '| |' if i == max_idx else '   '
        #         p_str += '{}{:3d}: {:8.4f} '.format(sym, a, v)
        #     print p_str

        self.cum_td_loss += update_info['td_loss'] if 'td_loss' in update_info \
                                                 and update_info['td_loss'] is not None else 0
        if not done:
            self.cum_reward += reward

        if done:
            print ("Episode {} done in {} steps, reward is {}, "
                   "average td_loss is {}. ").format(
                n_ep, n_steps, self.cum_reward,
                self.cum_td_loss / n_steps)
            total_steps += n_steps
            summary_proto.value.add(tag='n_steps', simple_value=n_steps)
            summary_proto.value.add(tag='total_steps', simple_value=total_steps)
            # summary_proto.value.add(tag='n_agent_steps', simple_value=n_agent_steps)
            summary_proto.value.add(
                tag='num_episode', simple_value=n_ep)
            summary_proto.value.add(
                tag='cum_reward', simple_value=self.cum_reward)
            summary_proto.value.add(
                tag='per_step_reward', simple_value=self.cum_reward / n_steps)
            summary_proto.value.add(
                tag='flag_success', simple_value=flag_success)
            self.reset()


class StepsSaver(object):
    def __init__(self, savedir):
        self.savedir = savedir
        self.eps_dir = None
        self.file = None
        self.stat_file = open(self.savedir + "/0000.txt", "w")

    def close(self):
        self.stat_file.close()

    def parse_state(self):
        return np.array(self.state)[:, :, 6:]

    def save(self, n_ep, n_step, state, action, vec_reward, reward,
                  done, cum_reward, flag_success):
        if self.file is None:
            self.eps_dir = self.savedir + "/" + str(n_ep).zfill(4)
            os.mkdir(self.eps_dir)
            self.file = open(self.eps_dir + "/0000.txt", "w")

        img_path = self.eps_dir + "/" + str(n_step + 1).zfill(4) + "_" + str(action) + ".jpg"
        cv2.imwrite(img_path, cv2.cvtColor(self.parse_state(), cv2.COLOR_RGB2BGR))
        self.file.write(str(n_step) + ',' + str(action) + ',' + str(reward) + '\n')
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
        stepsSaver = StepsSaver(FLAGS.savedir)
        logger = Logger()
        funcReward = FuncReward()
        while True:
            n_ep += 1
            n_steps = 0
            cum_reward = 0.0
            flag_success = False
            state = env.reset()
            while True:
                # Do act
                n_steps += 1
                total_steps += 1
                action = agent.act(state)
                next_state, vec_reward, done, info = env.step(action)
                # It has to explicitly trans n_skip and cnt_skip
                reward, done = funcReward.func(action, vec_reward, done, n_skip=agent._n_skip, cnt_skip=agent._cnt_skip)
                cum_reward += reward
                if done and reward > 0.0:
                    flag_success = True
                else:
                    flag_success = False
                update_info = agent.step(state, action, reward, next_state)
                # save intermediate info
                stepsSaver.save(n_ep, n_steps, state, action, vec_reward, reward,
                                    done, cum_reward, flag_success)
                logger.log(update_info, action, n_ep, total_steps, n_steps)

                state = next_state


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

