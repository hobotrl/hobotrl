import signal
import time
import traceback

import numpy as np
import tensorflow as tf

# Hobotrl
import hobotrl as hrl
from hobotrl.environments import FrameStack
# initialD
# from ros_environments.honda import DrivingSimulatorEnv
from playground.initialD.ros_environments.clients import DrivingSimulatorEnvClient as DrivingSimulatorEnv
# Gym
from gym.spaces import Discrete, Box
import cv2
import os
import resnet_pq
from pprint import pprint

# Environment
def func_compile_reward(rewards):
    return rewards


def func_compile_obs(obss):
    obs1 = obss[-1][0]
    # action = obss[-1][1]
    # print obss[-1][1]
    # print obs1.shape
    return obs1



def f_net(inputs):
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    # saver.restore(sess, checkpoint)
    state = inputs[0]
    print "global varibles: "
    pprint(tf.global_variables())
    print "========\n"*5
    res = resnet_pq.ResNet(3, name="train")
    pi = res.build_tower(state)
    q = res.build_new_tower(state)

    print "========\n"*5

    # pi = tf.nn.softmax(pi)
    # q = res.build_new_tower(state)
    # print "q type: ", type(q)
    # return {"q":q, "pi": pi}
    return {"pi": pi, "q": q}


ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']]


def func_compile_action(action):
    ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']]
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


def resize_state(state):
    return np.concatenate([cv2.resize(state[:, :, :3], (256, 256)),
                            cv2.resize(state[:, :, 3:6], (256, 256)),
                            cv2.resize(state[:, :, 6:9], (256, 256))], -1)

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


def record(summary_writer, step_n, info):
    for name in info:
        value = info[name]
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=np.mean(value))
        summary_writer.add_summary(summary, step_n)


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
stack_num = 3
env.observation_space = Box(low=0, high=255, shape=(256, 256, 9))
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
env.action_space = Discrete(len(ACTIONS))
env = FrameStack(env, 3)

n_interactive = 0
n_skip = 1
n_ep = 0  # last ep in the last run, if restart use 0
n_test = 10  # num of episode per test run (no exploration)
state_shape = (256, 256, 9)
tf.app.flags.DEFINE_string("logdir",
                           "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pq_rename_learn_q",
                           """save tmp model""")
tf.app.flags.DEFINE_string("savedir",
                           "/home/pirate03/hobotrl_data/playground/initialD/exp/"
                           "record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/resnet_pq_learn_q",
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
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

# sv = agent.init_supervisor(
#         graph=tf.get_default_graph(), worker_index=0,
#         init_op=tf.global_variables_initializer(), save_dir=FLAGS.logdir
#     )
summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=tf.get_default_graph())

os.mkdir(FLAGS.savedir)

not_restore_var_names = {'optimizers/beta1_power:0', 'optimizers/beta2_power:0',
                         'learn/q_logits/fc/weights:0', 'learn/q_logits/fc/biases:0',
                         'learn/q_logits/fc/weights/Adam:0', 'learn/q_logits/fc/weights/Adam_1:0',
                         'learn/q_logits/fc/biases/Adam:0', 'learn/q_logits/fc/biases/Adam_1:0',
                         'global_step:0'}

print "global_vars: "
pprint(tf.global_variables())


restore_var_list = []
for var in tf.global_variables():
    print "var_name: ", var.name
    if var.name in not_restore_var_names:
        pass
    else:
        restore_var_list.append(var)
print "restore vars name: "
pprint(restore_var_list)

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
            for i in range(stack_num-1):
                state, reward, done, info = env.step(0)

            print "========reset======\n" * 5
            while True:
                s1 = time.time()
                state = resize_state(np.array(state))
                action = agent.act(state=state, evaluate=False, sess=sess)
                next_state, vec_reward, done, info = env.step(action)
                next_state = resize_state(np.array(next_state))
                reward = func_compile_reward_agent(vec_reward, action)
                img = state[:, :, :3]
                img_path = eps_dir + "/" + str(n_steps + 1).zfill(4) + \
                           "_" + str(action) + ".jpg"
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                recording_file.write(str(n_steps) + ',' + str(action) + ',' + str(reward) + '\n')
                vec_reward = np.mean(np.array(vec_reward), axis=0)
                vec_reward = vec_reward.tolist()
                str_reward = ""
                for r in vec_reward:
                    str_reward += str(r)
                    str_reward += ","
                str_reward += "\n"
                recording_file.write(str_reward)
                info = agent.step(state=state, action=action, reward=reward, next_state=next_state,
                                  episode_done=done)
                record(summary_writer, n_steps, info)

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

