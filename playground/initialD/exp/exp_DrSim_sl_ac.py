import signal
import sys
import traceback

import os

sys.path.append('../../..')
sys.path.append('..')


import numpy as np

import tensorflow as tf

from playground.initialD.ros_environments.core import DrivingSimulatorEnv

from std_msgs.msg import Char, Bool, Int16, Float32
from sensor_msgs.msg import CompressedImage
import sklearn.metrics

import cv2

from exp.utils import resnet
import hobotrl as hrl


# Environment
def compile_reward(rewards):
    # print "reward 0: ", rewards[0]
    # print "reward 1: ", rewards[1]
    # print "reward 2: ", rewards[2]
    rewards = map(
        lambda reward: sum((
            -100.0 * float(reward[0]),  # obstacle 0 or -0.04
             -0.1 * float(reward[1])*(float(reward[1])>2.0),  # distance to 0.002 ~ 0.008
             0.2 * float(reward[2]),  # car_velo 0 ~ 0.08
        )),
        rewards)

    return np.mean(rewards)
    # return 1.0


def func_compile_action(action):
    ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']]
    return ACTIONS[action]

def compile_obs(obss):
    obs1 = obss[-1][0]
    # rule_action = obss[-1][1]
    # obs = np.concatenate([obs1, obs2, obs3], axis=2)
    return obs1

def evaluate(y_true, preds):
    prec = sklearn.metrics.precision_score(y_true, preds, average=None)
    rec = sklearn.metrics.recall_score(y_true, preds, average=None)
    f1 = sklearn.metrics.f1_score(y_true, preds, average=None)
    conf_mat = sklearn.metrics.confusion_matrix(y_true, preds)
    return prec, rec, f1, conf_mat
    # print "val_prec: {}".format(prec)
    # print "val_rec: {}".format(rec)
    # print "val_f1: {}".format(f1)
    # print "val_conf_mat: {}".format(conf_mat)


# tf.app.flags.DEFINE_string("train_dir", "./log_train_mix_all_and_s5_test_s5-1_2", """save tmp model""")
# tf.app.flags.DEFINE_string('checkpoint',
#     "/home/pirate03/PycharmProjects/resnet-18-tensorflow/log_mix_all_and_s5/model.ckpt-11999",
#                            """Model checkpoint to load""")

# FLAGS = tf.app.flags.FLAGS
#
#
# if not os.path.exists(FLAGS.train_dir):
#     os.mkdir(FLAGS.train_dir)
# else:
#     sys.exit(1)


def gen_backend_cmds():
    ws_path = '/home/lewis/Projects/catkin_ws_pirate03_lowres350_dynamic/'
    initialD_path = '/home/pirate03/PycharmProjects/hobotrl/playground/initialD/'
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
         utils_path+'honda_dynamic_obs_template.launch', 30],
        # 3. start roscore
        ['roscore'],
        # 4. start reward function script
        ['python', backend_path+'gazebo_rl_reward.py'],
        # 5. start simulation restarter backend
        ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
        # 6. [optional] video capture
        ['python', backend_path+'non_stop_data_capture.py', 0]
    ]
    return backend_cmds

env = DrivingSimulatorEnv(
    backend_cmds=gen_backend_cmds(),
    defs_obs=[('/training/image/compressed', CompressedImage)],
    func_compile_obs=compile_obs,
    defs_reward=[
        ('/rl/has_obstacle_nearby', Bool),
        ('/rl/distance_to_longestpath', Float32),
        ('/rl/car_velocity', Float32),
        ('/rl/last_on_opposite_path', Int16),
        ('/rl/on_pedestrian', Bool)],
    func_compile_reward=compile_reward,
    func_compile_action=func_compile_action,
    defs_action=[('/autoDrive_KeyboardMode', Char)],
    rate_action=10.0,
    window_sizes={'obs': 2, 'reward': 3},
    buffer_sizes={'obs': 2, 'reward': 3},
    step_delay_target=0.4
)

# def f_net(inputs):
#     l2 = 1e-4
#     state = inputs[0]
#     q = hrl.network.Utils.layer_fcs(state, [200, 100], 3,
#                                     l2=l2, var_scope="q")
#     pi = hrl.network.Utils.layer_fcs(state, [200, 100], 3,
#                                      activation_out=tf.nn.softmax, l2=l2, var_scope="pi")
#     return {"q": q, "pi": pi}

batch_size = 1
hp = resnet.HParams(batch_size=batch_size,
                    num_gpus=1,
                    num_classes=3,
                    weight_decay=0.001,
                    momentum=0.9,
                    finetune=True)


def f_net(inputs):
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    # saver.restore(sess, checkpoint)
    state = inputs[0]
    res = resnet.ResNet(hp, global_step, name="train")
    pi = res.build_origin_tower(state)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
    checkpoint = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/exp/rename_net/resnet"
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
    print "pi type: ", type(pi)
    # pi = tf.nn.softmax(pi)
    q = res.build_new_tower(state)
    print "q type: ", type(q)
    return {"q":q, "pi": pi}


def preprocess_image(input_image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (input_image - imagenet_mean) / imagenet_std
    return image


def record(summary_writer, step_n, info):
    for name in info:
        value = info[name]
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=np.mean(value))
        summary_writer.add_summary(summary, step_n)

state_shape = (224, 224, 3)

global_step = tf.get_variable(
            'learn/global_step', [], dtype=tf.int32,
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

# env.observation_space = Box(low=0, high=255, shape=(640, 640, 3))
# env.action_space = Discrete(3)
# env.reward_range = (-np.inf, np.inf)
# env.metadata = {}
# env = FrameStack(env, 1)
# state_shape = env.observation_space.shape
# graph = tf.get_default_graph()

n_interactive = 0
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)
reward_decay = 0.7
# filename = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/train.tfrecords"
# replay_buffer = initialD_input.init_replay_buffer(filename, replay_size=10000, batch_size=200)
logdir = "./tmp_DrSim_sl_ac"

try:
    # config = tf.ConfigProto()
    config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sv = agent.init_supervisor(
        graph=tf.get_default_graph(), worker_index=0,
        init_op=tf.global_variables_initializer(), save_dir=logdir
    )
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    with sv.managed_session(config=config) as sess:
        agent.set_session(sess)
        # print "========\n"*5
        # lr = graph.get_operation_by_name('lr').outputs[0]
        n_steps = 0
        while True:
            n_ep += 1
            cum_reward = 0.0
            cum_td_loss = 0.0
            cum_spg_loss = 0.0
            state = env.reset()
            print "========reset======\n"*5
            # img = cv2.resize(state, (224, 224)) / 255.0 - 0.5
            img = cv2.resize(state, (224, 224))
            img = preprocess_image(img)
            while True:
                # print "img shape: ", img.shape
                action = agent.act(state=img, evaluate=False, sess=sess)
                next_state, reward, done, info = env.step(action)
                # next_img = cv2.resize(next_state, (224, 224)) / 255.0 - 0.5
                next_img = cv2.resize(next_state, (224, 224))
                next_img = preprocess_image(next_img)
                n_steps += 1
                cum_reward = reward + reward_decay * cum_reward
                info = agent.step(state=img, action=action, reward=reward, next_state=next_img, episode_done=done)
                print "action: ", action
                record(summary_writer, n_steps, info)
                if done is True:
                    print "========Run Done=======\n"*5
                    break
                img = next_img

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

