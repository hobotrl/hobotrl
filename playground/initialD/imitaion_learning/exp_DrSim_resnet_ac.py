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
from playground.initialD.ros_environments import DrivingSimulatorEnv

import rospy
import message_filters
from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage

from gym.spaces import Discrete, Box
import cv2

from playground.initialD.imitaion_learning import initialD_input
import random
# from playground.resnet import resnet
import resnet
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
    rewards = map(
        lambda reward: sum((
            -10.0 * float(reward[0]),  # obstacle 0 or -0.04
            -0.1 * float(reward[1]) * (float(reward[1]) > 2.0),  # distance to 0.002 ~ 0.008
            0.2 * float(reward[2]),  # car_velo 0 ~ 0.08
        )),
        rewards)
    return np.mean(rewards)

def compile_obs(obss):
    obs1 = obss[-1][0]
    # obs = np.concatenate([obs1, obs2, obs3], axis=2)
    return obs1

def record(summary_writer, step_n, info):
    for name in info:
        value = info[name]
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=np.mean(value))
        summary_writer.add_summary(summary, step_n)

hp = resnet.HParams(batch_size=64,
                    num_gpus=1,
                    num_classes=3,
                    weight_decay=0.001,
                    momentum=0.9,
                    finetune=True)

def f_net(inputs):
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    # saver.restore(sess, checkpoint)
    state = inputs[0]
    print "global varibles: ", tf.global_variables()
    print "========\n"*5
    res = resnet.ResNet(hp, global_step, name="train")
    pi = res.build_origin_tower(state)
    q = res.build_new_tower(state)

    print "========\n"*5

    # pi = tf.nn.softmax(pi)
    # q = res.build_new_tower(state)
    # print "q type: ", type(q)
    # return {"q":q, "pi": pi}
    return {"pi":pi, "q":q}

tf.app.flags.DEFINE_string("logdir",
                           "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/DrSim_resnet_rename_pi_q_opt_add_logitsadam",
                           """save tmp model""")


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
    step_delay_target=0.5
)
env.observation_space = Box(low=0, high=255, shape=(640, 640, 3))
env.action_space = Discrete(3)
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
# env = FrameStack(env, 1)
ACTIONS = [(Char(ord(mode)),) for mode in ['s', 'd', 'a']]

state_shape = (224, 224, 3)
n_interactive = 0
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)
n_update = 0
reward_decay = 0.7

# replay_buffer = initialD_input.init_replay_buffer(filename, replay_size=10000, batch_size=200)

global_step = tf.get_variable(
            'learn/global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
agent = hrl.ActorCritic(
            f_create_net=f_net,
            state_shape=state_shape,
            # ACUpdate arguments
            discount_factor=0.7,
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

config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

sv = agent.init_supervisor(
        graph=tf.get_default_graph(), worker_index=0,
        init_op=tf.global_variables_initializer(), save_dir=FLAGS.logdir
    )
summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=tf.get_default_graph())

try:
    # config = tf.ConfigProto()
    with sv.managed_session(config=config) as sess:
        agent.set_session(sess)
        n_steps = 0
        while True:
            n_ep += 1
            cum_reward = 0.0
            cum_td_loss = 0.0
            cum_spg_loss = 0.0
            all_scenes = []
            state = env.reset()
            print "========reset======\n" * 5
            img = cv2.resize(state, (224, 224))
            pr_img = initialD_input.preprocess_image(img)
            while True:
                action = agent.act(state=pr_img, evaluate=False, sess=sess)
                print "action: ", action
                all_scenes.append([np.copy(img), action])
                next_state, reward, done, info = env.step(ACTIONS[action])
                next_img = cv2.resize(next_state, (224, 224))
                pr_next_img = initialD_input.preprocess_image(next_img)
                info = agent.step(state=pr_img, action=action, reward=reward, next_state=pr_next_img, episode_done=done)
                record(summary_writer, n_steps, info)
                n_steps += 1
                if done is True:
                    print "========Run Done=======\n" * 5
                    break
                img = next_img
                pr_img = pr_next_img

            for i, ele in enumerate(all_scenes):
                cv2.imwrite(FLAGS.logdir + "/"
                            + "ac/"
                            + str(n_ep) + "_" +
                                str(i) + "_" + str(ele[1]) + ".jpg", ele[0])

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

