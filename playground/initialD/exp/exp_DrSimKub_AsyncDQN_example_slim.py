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

# Tensorflow
import tensorflow as tf

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
gamma = 0.9
greedy_epsilon = CappedLinear(50000, 0.5, 0.05)
# --- replay buffer
replay_capacity = 300000
replay_bucket_size = 100
replay_ratio_active = 0.05
replay_max_sample_epoch = 2
replay_upsample_bias = (1, 1, 1, 0.1)
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
    sv = tf.train.Supervisor(
        graph=tf.get_default_graph(),
        is_chief=True,
        init_op=tf.global_variables_initializer(),
        logdir=tf_log_dir,
        save_summaries_secs=10,
        save_model_secs=3600)
    with sv.managed_session(config=config) as sess, \
         AsynchronousAgent(
             agent=_agent, method='rate', rate=update_rate) as agent:
        agent.set_session(sess)
        sess.run(op_set_lr, feed_dict={lr_in: learning_rate})
        logging.warning(
            "Using learning rate {}".format(sess.run(lr))
        )
        n_ep = 0
        n_env_steps = 0
        n_agent_steps = 0
        while True:
            cum_reward = 0.0
            n_ep_steps = 0
            state  = env.reset()
            while True:
                action = agent.act(state)
                print_qvals(
                    n_env_steps, _agent, state, action, AGENT_ACTIONS
                )
                next_state, reward, done, env_info = env.step(action)
                agent_info = agent.step(
                    sess=sess, state=state, action=action,
                    reward=reward, next_state=next_state,
                    episode_done=done
                )
                state = next_state
                n_agent_steps += 1
                n_env_steps += 1
                n_ep_steps += 1
                cum_reward += reward
                summary_proto = log_info(
                    agent_info, env_info,
                    done,
                    cum_reward,
                    n_ep, n_ep_steps, n_env_steps, n_agent_steps
                )
                sv.summary_computed(sess, summary=summary_proto)
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

