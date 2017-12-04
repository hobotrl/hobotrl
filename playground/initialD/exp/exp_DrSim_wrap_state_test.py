# -*- coding: utf-8 -*-
"""Experiment script example lane decision.
:author: Jingchu Liu
"""
# Basics
import os
import signal
import logging
import sys
import traceback
# CV and NP
import numpy as np
import cv2
# Tensorflow
import tensorflow as tf
from tensorflow.python.training.summary_io import SummaryWriterCache
# Hobotrl
sys.path.append('../../..')  # hobotrl home
from hobotrl.algorithms import DQN
from hobotrl.network import LocalOptimizer
from hobotrl.environments import FrameStack
from hobotrl.sampling import TransitionSampler
from hobotrl.playback import BalancedMapPlayback, BigPlayback
from hobotrl.async import AsynchronousAgent
from hobotrl.utils import CappedLinear
# initialD
sys.path.append('..')  # initialD home
from ros_environments.clients import DrSimDecisionK8S
from exp.utils.func_networks import f_dueling_q
from exp.utils.wrappers import EnvNoOpSkipping, EnvRewardVec2Scalar
from exp.utils.logging_fun import print_qvals, log_info

# ============= Set Parameters =================
# -------------- Env
n_skip = 6
n_stack = 3
if_random_phase = True
# -------------- Agent
# --- agent basic
ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
AGENT_ACTIONS = ALL_ACTIONS[:3]
num_actions = len(AGENT_ACTIONS)
gamma = 0.9  # discount factor
greedy_epsilon = CappedLinear(50000, 0.5, 0.05)  # exploration rate accroding to step
# --- replay buffer
replay_capacity = 15000 # in MB, maxd buf at disk. i step about 1MB
replay_bucket_size = 100  # how many step in a block
replay_ratio_active = 0.1  # ddr ratio
replay_max_sample_epoch = 2  # max replay times
replay_upsample_bias = (1, 1, 1, 0.1)  # upsample for replay buf redistribution, accroding to ?
# --- NN architecture
f_net = lambda inputs: f_dueling_q(inputs, num_actions)
if_ddqn = True
# --- optimization
batch_size = 8  # mini batch
learning_rate = 1e-4
target_sync_interval = 1  # lay update?
target_sync_rate = 1e-4  # para for a filter which is similar to lazy update
update_interval = 1
max_grad_norm = 1.0  # limit max gradient
sample_mimimum_count = 1000  # what?
update_rate = 4.0  # updates per second by the async wrapper
# --- logging and ckpt
tf_log_dir = "./experiment"
replay_cache_dir = "./ReplayBufferCache/experiment"
# ==========================================

# Environment
env = EnvNoOpSkipping(
    env=EnvRewardVec2Scalar(FrameStack(DrSimDecisionK8S(), n_stack)),
    n_skip=n_skip, gamma=gamma, if_random_phase=if_random_phase
)

# ==========================================
# State Wrapper

src_size = (350,350)
dst_size = (350,350)
center_src = (175,175)
center_dst = (175,175)
linear_part_ratio_dst = 0.2
k_scale = 1.0  # dst_size[0]/src_size[0] typically, set by hand if needed
d = 1.0 / k_scale

mapx = np.zeros((dst_size[1], dst_size[0]), dtype=np.float32)
mapy = np.zeros((dst_size[1], dst_size[0]), dtype=np.float32)


def remap_core(x, size_s, size_d, c_src, c_dst, lr):
    lp = c_dst - c_dst * lr
    lp_src = c_src - c_dst * lr
    hp = c_dst + (size_d - c_dst) * lr
    hp_src = c_src + (size_d - c_dst) * lr
    a1 = -(lp_src - d * lp) / (lp * lp)  # -(lp_src-lp) / (lp*lp)
    b1 = d - 2 * a1 * lp  # add d
    # a2      = (hp_src-hp - size_s + size_d) / (-(hp-size_d)*(hp-size_d))
    a2 = (hp_src - d * hp - size_s + d * size_d) / (-(hp - size_d) * (hp - size_d))  # add d
    b2 = d - 2 * a2 * hp  # add d, 1-2a*hp
    c2 = hp_src - a2 * hp * hp - b2 * hp
    if x < lp:
        y = a1 * x * x + b1 * x
    elif x < hp:
        y = x + (c_src - c_dst)
    else:
        y = a2 * x * x + b2 * x + c2
    return y

def fx(x):
    return remap_core(x, src_size[0], dst_size[0], center_src[0], center_dst[0], linear_part_ratio_dst)

def fy(y):
    return remap_core(y, src_size[1], dst_size[1], center_src[1], center_dst[1], linear_part_ratio_dst)

for x in range(dst_size[0]):
    tmp = fx(x)
    for y in range(dst_size[1]):
        mapx[y][x] = tmp
for y in range(dst_size[1]):
    tmp = fy(y)
    for x in range(dst_size[0]):
        mapy[y][x] = tmp

# normalize map to the src image size, d(srctodst) will be affected by ratio
map_max = mapx.max()
map_min = mapx.min()
ratio = (src_size[0] - 1) / (map_max - map_min)
mapx = ratio * (mapx - map_min)
map_max = mapy.max()
map_min = mapy.min()
ratio = (src_size[1] - 1) / (map_max - map_min)
mapy = ratio * (mapy - map_min)


def remap_process(frame):
    # remap
    dst = cv2.remap(np.asarray(frame), mapx, mapy, cv2.INTER_LINEAR)
    # for display
    last_frame = dst[:, :, 0:3]
    cv2.imshow("image1", cv2.resize(last_frame, (320, 320), interpolation=cv2.INTER_LINEAR))
    cv2.waitKey(10)
    return dst

# ===============






replay_buffer = None
try:
    graph = tf.get_default_graph()
    lr = tf.get_variable(
        'learning_rate', [], dtype=tf.float32,
        initializer=tf.constant_initializer(1e-3), trainable=False
    )
    lr_in = tf.placeholder(dtype=tf.float32)
    op_set_lr = tf.assign(lr, lr_in)
    optimizer_td = tf.train.AdamOptimizer(learning_rate=lr)
    global_step = tf.get_variable(
        'global_step', [], dtype=tf.int32,
        initializer=tf.constant_initializer(0), trainable=False)

    replay_buffer = BigPlayback(
        bucket_cls=BalancedMapPlayback,
        cache_path=replay_cache_dir,
        capacity=replay_capacity,
        bucket_size=replay_bucket_size,
        ratio_active=replay_ratio_active,
        max_sample_epoch=replay_max_sample_epoch,
        num_actions=num_actions,
        upsample_bias=replay_upsample_bias
    )
    state_shape = env.observation_space.shape
    _agent = DQN(
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with _agent.create_session(config=config, save_dir=tf_log_dir,
            save_checkpoint_secs=3600, save_summaries_secs=5) as sess, \
         AsynchronousAgent(
             agent=_agent, method='rate', rate=update_rate) as agent:
        summary_writer = SummaryWriterCache.get(tf_log_dir)
        sess.run(op_set_lr, feed_dict={lr_in: learning_rate})
        logging.warning(
            "Using learning rate {}".format(sess.run(lr))
        )
        n_ep = 0
        n_total_steps = 0
        while True:
            cum_reward = 0.0
            n_ep_steps = 0
            state = env.reset()
            state = remap_process(state)
            while True:
                action = agent.act(state)
                print_qvals(
                    n_ep_steps, _agent, state, action, AGENT_ACTIONS
                )
                next_state, reward, done, env_info = env.step(action)
                state = remap_process(state)
                agent_info = agent.step(
                    sess=sess, state=state, action=action,
                    reward=reward, next_state=next_state,
                    episode_done=done
                )
                state = next_state
                n_total_steps += 1
                n_ep_steps += 1
                cum_reward += reward
                summary_proto = log_info(
                    agent_info, env_info,
                    done,
                    cum_reward,
                    n_ep, n_ep_steps, n_total_steps
                )
                summary_writer.add_summary(summary_proto, n_total_steps)
                if done:
                    n_ep += 1
                    logging.warning(
                        "Episode {} finished in {} steps, reward is {}.".format(
                            n_ep, n_ep_steps, cum_reward,
                        )
                    )
                    break
except Exception as e:
    print e.message
    traceback.print_exc()
finally:
    print "="*30
    print "Tidying up..."
    # kill orphaned monitor daemon process
    env.env.exit()
    if replay_buffer is not None:
        replay_buffer.close()
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    print "="*30

