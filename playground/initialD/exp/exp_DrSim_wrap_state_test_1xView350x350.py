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
import zmq
import dill
import wrapt
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
#from ros_environments.clients import DrSimDecisionK8S
from exp.utils.func_networks import f_dueling_q
from exp.utils.wrappers import EnvNoOpSkipping, EnvRewardVec2Scalar
from exp.utils.logging_fun import print_qvals, log_info
#ros_environments
sys.path.append('../ros_environments/')
from hobotrl.environments.kubernetes.client import KubernetesEnv
from server import DrSimDecisionK8SServer
# Gym
from gym.spaces import Discrete, Box

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
replay_capacity = 20000 # in MB, maxd buf at disk. i step about 1MB
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


class DrivingSimulatorEnvClient(object):
    def __init__(self, address, port, **kwargs):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(address, port))
        kwargs_send = {}
        for key, value in kwargs.iteritems():
            kwargs_send[key] = dill.dumps(value)
        self.socket.send_pyobj(('start', kwargs_send))
        msg_type, msg_payload = self.socket.recv_pyobj()
        if not msg_type == 'start':
            raise Exception('EnvClient: msg_type is not start.')

    def reset(self):
        self.socket.send_pyobj(('reset', None))
        msg_type, msg_payload = self.socket.recv_pyobj()
        if not msg_type == 'reset':
            raise Exception('EnvClient: msg_type is not reset.')
        return msg_payload

    def step(self, action):
        self.socket.send_pyobj(('step', (action,)))
        msg_type, msg_payload = self.socket.recv_pyobj()
        if not msg_type == 'step':
            raise Exception('EnvClient: msg_type is not step.')
        return msg_payload

    def exit(self):
        self.socket.send_pyobj(('exit', None))
        self.socket.close()
        # self.context.term()
        return

    def close(self):
        self.exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()


class DrSimDecisionK8S(wrapt.ObjectProxy):
    _version = '20171127'
    _ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
    def __init__(self, image_uri=None, backend_cmds=None, *args, **kwargs):
        # Simulator Docker image to use
        if image_uri is None:
            _image_uri = "docker.hobot.cc/carsim/simulator_gpu_kub:latest"
        else:
            _image_uri = image_uri

        # bash commands to be executed in each episode to start simulation
        # backend
        if backend_cmds is None:
            _backend_cmds = self.gen_default_backend_cmds()
        else:
            _backend_cmds = backend_cmds

        # ROS topic definition tuples for observation, reward, and action
        _defs_obs = [
            ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
            ('/decision_result', 'std_msgs.msg.Int16'),
            ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
        ]
        _defs_reward = [
            ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
            ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
            ('/rl/obs_factor', 'std_msgs.msg.Float32'),
            ('/rl/current_road_validity', 'std_msgs.msg.Int16'),
            ('/rl/entering_intersection', 'std_msgs.msg.Bool'),
            ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
            ('/rl/on_biking_lane', 'std_msgs.msg.Bool'),
            ('/rl/on_innerest_lane', 'std_msgs.msg.Bool'),
            ('/rl/on_outterest_lane', 'std_msgs.msg.Bool')
        ]
        _defs_action = [('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')]

        _func_compile_obs = DrSimDecisionK8SServer.func_compile_obs
        _func_compile_reward = DrSimDecisionK8SServer.func_compile_reward
        _func_compile_action = DrSimDecisionK8SServer.func_compile_action

        # Build wrapped environment, expose step() an reset()
        _env = KubernetesEnv(
            image_uri=_image_uri,
            remote_client_env_class=DrivingSimulatorEnvClient,
            backend_cmds=_backend_cmds,
            defs_obs=_defs_obs,
            defs_reward=_defs_reward,
            defs_action=_defs_action,
            rate_action=10.0,
            window_sizes={'obs': 3, 'reward': 3},
            buffer_sizes={'obs': 3, 'reward': 3},
            func_compile_obs=_func_compile_obs,
            func_compile_reward=_func_compile_reward,
            func_compile_action=_func_compile_action,
            step_delay_target=0.5
        )
        super(DrSimDecisionK8S, self).__init__(_env)

        # Gym env required attributes
        self.observation_space = Box(low=0, high=255, shape=(350, 350, 3))
        self.reward_range = Box(
            low=-np.inf, high=np.inf, shape=(len(_defs_reward),)
        )
        self.action_space = Discrete(len(self._ALL_ACTIONS))
        self.metadata = {}

    @staticmethod
    def gen_default_backend_cmds():
        ws_path = '/Projects/catkin_ws/'
        initialD_path = '/Projects/hobotrl/playground/initialD/'
        backend_path = initialD_path + 'ros_environments/backend_scripts/'
        utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
        backend_cmds = [
            # Parse maps
            ['python', utils_path + 'parse_map.py',
             ws_path + 'src/Map/src/map_api/data/honda_wider.xodr',
             utils_path + 'road_segment_info.txt'],
            # Generate obstacle configuration and write to launch file
            ['python', utils_path+'gen_launch_dynamic_v1.py',
             utils_path+'road_segment_info.txt', ws_path,
             utils_path+'state_remap_test.launch', 32, '--random_n_obs'],
            # Start roscore
            ['roscore'],
            # Reward function script
            ['python', backend_path + 'gazebo_rl_reward.py'],
            # Road validity node script
            ['python', backend_path + 'road_validity.py',
             utils_path + 'road_segment_info.txt.signal'],
            # Simulation restarter backend
            ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
            # Video capture
            ['python', backend_path+'non_stop_data_capture.py']
        ]
        return backend_cmds

    def exit(self):
        self.__wrapped__.exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()
# ================


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
    dst = np.asarray(frame)  # cv2.remap(np.asarray(frame), mapx, mapy, cv2.INTER_LINEAR)
    # for display
    last_frame = dst[:, :, 0:3]
    cv2.imshow("image1", last_frame)
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
            #state = remap_process(state)
            while True:
                action = agent.act(state)
                print_qvals(
                    n_ep_steps, _agent, state, action, AGENT_ACTIONS
                )
                next_state, reward, done, env_info = env.step(action)
                #state = remap_process(state)
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

