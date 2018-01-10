# -*- coding: utf-8 -*-
"""Experiment script example lane decision.
:author: Jingchu Liu
"""
# Basics
import os
import signal
import sys
import logging
import traceback
# Data
import numpy as np
# Tensorflow
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
# Hobotrl
sys.path.append('../../..')  # hobotrl home
from hobotrl.algorithms import DQN
from hobotrl.network import LocalOptimizer
from hobotrl.environments import FrameStack
from hobotrl.sampling import TransitionSampler
from hobotrl.playback import MapPlayback, BalancedMapPlayback, BigPlayback
from hobotrl.async import AsynchronousAgent
from hobotrl.utils import CappedLinear
# initialD
sys.path.append('..')  # initialD home
from ros_environments.clients import DrSimDecisionK8S
from exp.utils.func_networks import f_dueling_q
# from exp.utils.skipping_masking import NonUniformSkip as SkippingAgent
from exp.utils.skipping_masking import RandFirstSkip as SkippingAgent
from exp.utils.logging_fun import StepsSaver, print_qvals, log_info

# ==============================================
# =========== Set Parameters Below =============
# ==============================================
# === Env
n_skip = 1
n_stack = 3
if_random_phase = True
# === Agent
# --- agent basic
ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
AGENT_ACTIONS = ALL_ACTIONS[:3]
num_actions = len(AGENT_ACTIONS)
noop = 3
gamma = 0.9
ckpt_step = 0
greedy_epsilon = CappedLinear(int(3e4)-ckpt_step, 0.2-(0.15/3e4*ckpt_step), 0.05)
start_step = ckpt_step
# --- replay buffer
replay_capacity = 300000
replay_bucket_size = 100
replay_ratio_active = 0.01
replay_max_sample_epoch = 2
# replay_upsample_bias = (1, 1, 1, 0.1)
# --- NN architecture
f_net = lambda inputs: f_dueling_q(inputs, num_actions)
if_ddqn = True
# --- optimization
batch_size = 8
learning_rate = 1e-4
target_sync_interval = 1
target_sync_rate = 1e-3
update_interval = 1
max_grad_norm = 1.0
sample_mimimum_count = 100
update_ratio = 8.0/6.0
# --- logging and ckpt

tf.app.flags.DEFINE_bool(
    "test", False,
    "Test or not.")
tf.app.flags.DEFINE_string(
    "dir_prefix", "./exp09/3",
    "Prefix for model ckpt and event file.")
tf.app.flags.DEFINE_string(
    "tf_log_dir", "ckpt",
    "Path for model ckpt and event file.")
tf.app.flags.DEFINE_string(
    "our_log_dir", "logging",
    "Path for our logging data.")
tf.app.flags.DEFINE_string(
    "replay_cache_dir", "ReplayBufferCache",
    "Replay buffer cache path.")
tf.app.flags.DEFINE_float(
    "gpu_mem_fraction", 0.15,
    "Replay buffer cache path.")
tf.app.flags.DEFINE_float(
    "save_checkpoint_secs", 3600,
    "Seconds to save tf model check points.")

FLAGS = tf.app.flags.FLAGS

if FLAGS.test:
    replay_capacity = 300
    replay_ratio_active = 1.0
else:
    replay_capacity = 300000
    replay_ratio_active = 0.01

# ===  Reward function
class FuncReward(object):
    def __init__(self, gamma):
        self.__gamma = gamma
        self._ema_speed = 10.0
        self._ema_dist = 0.0
        self._obs_risk = 0.0
        self._road_change = False
        self._mom_opp = 0.0
        self._mom_biking = 0.0
        self._steering = False
        self._waiting_steps = 0

    def reset(self):
        self._ema_speed = 10.0
        self._ema_dist = 0.0
        self._obs_risk = 0.0
        self._road_change = False
        self._mom_opp = 0.0
        self._mom_biking = 0.0
        self._steering = False

    def _func_scalar_reward(self, rewards, action):
        """Coverts a vector reward into a scalar."""
        info = {}

        # append a reward that is 1 when action is lane switching
        rewards = rewards.tolist()
        print (' '*3 + 'R: [' + '{:4.2f} ' * len(rewards) + ']').format(
            *rewards),

        # extract relevant rewards.
        speed = rewards[0]
        dist = rewards[1]
        obs_risk = rewards[2]
        # road_invalid = rewards[3] > 0.01  # any yellow or red
        road_change = rewards[4] > 0.01  # entering intersection
        opp = rewards[5]
        biking = rewards[6]
        # inner = rewards[7]
        # outter = rewards[8]
        steer = np.logical_or(action == 1, action == 2)

        if speed < 0.1:
            self._waiting_steps += 1
        else:
            self._waiting_steps = 0

        # update reward-related state vars
        ema_speed = 0.5 * self._ema_speed + 0.5 * speed
        ema_dist = 1.0 if dist > 2.0 else 0.9 * self._ema_dist
        mom_opp = min((opp < 0.5) * (self._mom_opp + 1), 20)
        mom_biking = min((biking > 0.5) * (self._mom_biking + 1), 12)
        steering = steer if action != 3 else self._steering
        self._ema_speed = ema_speed
        self._ema_dist = ema_dist
        self._obs_risk = obs_risk
        self._road_change = road_change
        self._mom_opp = mom_opp
        self._mom_biking = mom_biking
        self._steering = steering
        print '{:3.0f}, {:3.0f}, {:4.2f}, {:3.0f}'.format(
            mom_opp, mom_biking, ema_dist, self._steering),
        info['reward_fun/speed'] = speed
        info['reward_fun/dist2longest'] = dist
        info['reward_fun/obs_risk'] = obs_risk
        info['reward_fun/road_change'] = road_change
        info['reward_fun/on_opposite'] = opp
        info['reward_fun/on_biking'] = biking
        info['reward_fun/steer'] = steer
        info['reward_fun/mom_opposite'] = mom_opp
        info['reward_fun/mom_biking'] = mom_biking
        info['waiting_steps'] = self._waiting_steps

        # calculate scalar reward
        reward = [
            # velocity
            speed * 10 - 10,
            # obs factor
            -100.0 * obs_risk,
            # opposite
            -20 * (0.9 + 0.1 * mom_opp) * (mom_opp > 1.0),
            # ped
            -40 * (0.9 + 0.1 * mom_biking) * (mom_biking > 1.0),
            # steer
            steering * -40.0,
        ]
        reward = np.sum(reward) / 100.0
        print ': {:5.2f}'.format(reward)

        return reward, info

    def _func_early_stopping(self):
        """Several early stopping criterion."""
        info = {}
        done = False
        # switched lane while going into intersection.
        if self._road_change and self._ema_dist > 0.2:
            print "[Episode early stopping] turned into intersection."
            done = True
            info['banned_road_change'] = True

        # used biking lane to cross intersection
        if self._road_change and self._mom_biking > 0:
            print "[Episode early stopping] entered intersection on biking lane."
            done = True
            info['banned_road_change'] = True

        # hit obstacle
        if self._obs_risk > 1.0:
            print "[Episode early stopping] hit obstacle."
            done = True

        # waiting too long
        if FLAGS.test and self._waiting_steps > 80:
            print "[Episode early stopping] waiting too long"
            done = True

        return done, info

    def _func_skipping_bias(self, reward, done, info, n_skip, cnt_skip):
        new_info = {}
        if 'banned_road_change' in info:
            reward -= 1.0 * (n_skip - cnt_skip)
        if done:
            pass
            #reward /= (1 - self.__gamma) / (n_skip - cnt_skip)
        new_info['reward_fun/reward'] = reward
        return reward, new_info

    def __call__(self, action, rewards, done, n_skip=1, cnt_skip=0):
        info = {}
        reward, info_diff  = self._func_scalar_reward(rewards, action)
        info.update(info_diff)
        early_done, info_diff = self._func_early_stopping()
        done = done | early_done
        info.update(info_diff)
        reward, info_diff = self._func_skipping_bias(
            reward, done, info, n_skip, cnt_skip)
        info.update(info_diff)
        if done:
            info['flag_success'] = reward > 0.0
            self.reset()

        return reward, done, info
# ==========================================
# ==========================================
# ==========================================

env, replay_buffer, _agent = None, None, None
try:
    # Parse flags
    # FLAGS = tf.app.flags.FLAGS
    dir_prefix = FLAGS.dir_prefix
    tf_log_dir = os.sep.join([dir_prefix, FLAGS.tf_log_dir])
    our_log_dir = os.sep.join([dir_prefix, FLAGS.our_log_dir])
    replay_cache_dir = os.sep.join([dir_prefix, FLAGS.replay_cache_dir])
    save_checkpoint_secs = FLAGS.save_checkpoint_secs
    gpu_mem_fraction = FLAGS.gpu_mem_fraction

    # Modify tf graph
    graph = tf.get_default_graph()
    # -- create learning rate var and optimizer
    lr = tf.get_variable(
        'learning_rate', [], dtype=tf.float32,
        initializer=tf.constant_initializer(1e-3), trainable=False
    )
    lr_in = tf.placeholder(dtype=tf.float32)
    op_set_lr = tf.assign(lr, lr_in)
    optimizer_td = tf.train.AdamOptimizer(learning_rate=lr)
    # -- create global step variable
    global_step = tf.get_variable(
        'global_step', [], dtype=tf.int32,
        initializer=tf.constant_initializer(0), trainable=False)

    def gen_default_backend_cmds():
        ws_path = '/Projects/catkin_ws/'
        initialD_path = '/Projects/hobotrl/playground/initialD/'
        backend_path = initialD_path + 'ros_environments/backend_scripts/'
        utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
        backend_cmds = [
            ['python', utils_path + '/iterate_test_case.py'],
            # Parse maps
            ['python', utils_path + 'parse_map.py',
             ws_path + 'src/Map/src/map_api/data/honda_wider.xodr',
             utils_path + 'road_segment_info.txt'],
            # Start roscore
            ['roscore'],
            # Reward function script
            ['python', backend_path + 'gazebo_rl_reward.py'],
            # Road validity node script
            ['python', backend_path + 'road_validity.py',
             utils_path + 'road_segment_info.txt.signal'],
            # Simulation restarter backend
            ['python', backend_path+'rviz_restart.py', 'next.launch'],
            # Video capture
            # ['python', backend_path+'non_stop_data_capture.py'],
        ]
        return backend_cmds

    # Environment
    if tf.test:
        env = FrameStack(DrSimDecisionK8S(backend_cmds=gen_default_backend_cmds()), n_stack)
    else:
        env = FrameStack(DrSimDecisionK8S(), n_stack)

    # Agent
    replay_buffer = BigPlayback(
        bucket_cls=MapPlayback,
        cache_path=replay_cache_dir,
        capacity=replay_capacity,
        bucket_size=replay_bucket_size,
        ratio_active=replay_ratio_active,
        max_sample_epoch=replay_max_sample_epoch,
    )
    state_shape = env.observation_space.shape
    __agent = DQN(
        f_create_q=f_net, state_shape=state_shape,
        # OneStepTD arguments
        num_actions=num_actions, discount_factor=gamma, ddqn=if_ddqn,
        # target network sync arguments
        target_sync_interval=target_sync_interval,
        target_sync_rate=target_sync_rate,
        # epsilon greedy arguments
        greedy_epsilon=greedy_epsilon,
        # optimizer arguments
        network_optimizer=LocalOptimizer(optimizer_td, max_grad_norm),
        # sampler arguments
        sampler=TransitionSampler(
            replay_buffer,
            batch_size=batch_size,
            interval=update_interval,
            minimum_count=sample_mimimum_count),
        # checkpoint
        global_step=global_step
     )
    # Utilities
    stepsSaver = StepsSaver(our_log_dir)
    reward_vector2scalar = FuncReward(gamma)
    # Configure sess
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
    with __agent.create_session(
            config=config, save_dir=tf_log_dir,
            save_checkpoint_secs=save_checkpoint_secs) as sess, \
        AsynchronousAgent(
            agent=__agent, method='ratio', ratio=update_ratio) as _agent:
        agent = SkippingAgent(
            # n_skip_vec=(2, 6, 6),
            agent=_agent, n_skip=n_skip, specific_act=noop
        )
        summary_writer = SummaryWriterCache.get(tf_log_dir)
        # set vars
        sess.run(op_set_lr, feed_dict={lr_in: learning_rate})
        print "Using learning rate {}".format(sess.run(lr))
        n_ep = 0
        n_total_steps = start_step
        # GoGoGo
        while n_total_steps <= 2.5e5:
            cum_reward = 0.0
            n_ep_steps = 0
            state = env.reset()
            while True:
                action = agent.act(state, exploration=not tf.test)
                if action != 3:
                    print_qvals(
                        n_ep_steps, __agent, state, action, AGENT_ACTIONS
                    )
                next_state, vec_reward, done, env_info = env.step(action)
                reward, done, reward_info = reward_vector2scalar(
                    action, vec_reward, done, agent.n_skip, agent.cnt_skip
                )
                agent_info = agent.step(
                    sess=sess, state=state, action=action,
                    reward=reward, next_state=next_state,
                    episode_done=done, learning_off=FLAGS.test
                )
                env_info.update(reward_info)
                summary_proto = log_info(
                    agent_info, env_info,
                    done,
                    cum_reward,
                    n_ep, n_ep_steps, n_total_steps,
                )
                summary_writer.add_summary(summary_proto, n_total_steps)
                n_total_steps += 1
                n_ep_steps += 1
                cum_reward += reward
                flag_success = reward_info['flag_success'] \
                    if 'flag_success' in reward_info else False
                stepsSaver.save(
                    n_ep, n_total_steps,
                    state, action, vec_reward, reward, done,
                    cum_reward, flag_success
                )
                state = next_state
                if done:
                    n_ep += 1
                    logging.warning(
                        "Episode {} finished in {} steps, reward is {}.".format(
                            n_ep, n_ep_steps, cum_reward,
                        )
                    )
                    break
            if FLAGS.test and n_ep >= 100:
                break

except Exception as e:
    print e.message
    traceback.print_exc()
finally:
    logging.warning("="*30)
    logging.warning("="*30)
    logging.warning("Tidying up...")
    # kill orphaned monitor daemon process
    if env is not None:
        env.env.exit()
    replay_buffer.close()
    if replay_buffer is not None:
        replay_buffer.close()
    if _agent is not None:
        _agent.stop()
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    logging.warning("="*30)

