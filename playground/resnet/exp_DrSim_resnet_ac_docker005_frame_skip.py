import signal
import time
import traceback

import numpy as np
import tensorflow as tf

# Hobotrl
import sys
sys.path.append('../')
sys.path.append('../../')
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
from tensorflow.python.training.summary_io import SummaryWriterCache


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
    if FLAGS.is_learn_q:
        pi = res.build_tower(state, stop_conv=True, stop_fc=True)
        q = res.build_new_tower(state, stop_conv=True)
    else:
        if FLAGS.is_fine_tune:
            pi = res.build_tower(state, stop_conv=True, stop_fc=False)
            q = res.build_new_tower(state, stop_conv=True)
        else:
            pi = res.build_tower(state, stop_conv=False, stop_fc=False)
            q = res.build_new_tower(state, stop_conv=False)

    print "========\n"*5

    # pi = tf.nn.softmax(pi)
    # q = res.build_new_tower(state)
    # print "q type: ", type(q)
    # return {"q":q, "pi": pi}
    return {"pi": pi, "q": q}


ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]


def func_compile_action(action):
    ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0, )]
    return ACTIONS[action]


# def func_compile_reward_agent(rewards, action=0):
#     global momentum_ped
#     global momentum_opp
#     rewards = np.mean(np.array(rewards), axis=0)
#     rewards = rewards.tolist()
#     rewards.append(np.logical_or(action==1, action==2))
#     print (' '*10+'R: ['+'{:4.2f} '*len(rewards)+']').format(*rewards),
#
#     # obstacle
#     rewards[0] *= 0.0
#     # distance to
#     # rewards[1] *= -10.0*(rewards[1]>2.0)
#     rewards[1] *= 0.0
#     # velocity
#     rewards[2] *= 10
#     # opposite
#     momentum_opp = (rewards[3]<0.5)*(momentum_opp+(1-rewards[3]))
#     momentum_opp = min(momentum_opp, 20)
#     rewards[3] = -20*(0.9+0.1*momentum_opp)*(momentum_opp>1.0)
#     min reward[3] == -0.6
#     # ped
#     momentum_ped = (rewards[4]>0.5)*(momentum_ped+rewards[4])
#     momentum_ped = min(momentum_ped, 12)
#     rewards[4] = -40*(0.9+0.1*momentum_ped)*(momentum_ped>1.0)
#     min reward[4] == -0.8
#     # obs factor
#     rewards[5] *= -100.0
#     # steering
#     rewards[6] *= -10.0
#     reward = np.sum(rewards)/100.0
#     print '{:6.4f}, {:6.4f}'.format(momentum_opp, momentum_ped),
#     print ': {:7.4f}'.format(reward)
#     return reward


def func_compile_reward_agent(rewards):
    """
    :param rewards: rewards[0]: obstacle???
                    rewards[1]: distance_to_planning_line
                    rewards[2]: velocity, 0-8.5
                    rewards[3]: oppsite reward, 0.0 if car is on oppsite else 1.0
                    rewards[4]: pedestrain reward, 1.0 if car is on pedestrain else 0.0
                    rewards[5]: obs factor???
    :param action:
    :return:
    """
    rewards = np.mean(np.array(rewards), axis=0)
    print (' ' * 10 + 'R: [' + '{:4.2f} ' * len(rewards) + ']').format(*rewards),
    # if car is on opp side or car is on ped side, get reward of -1.0
    if rewards[3] < 0.5 or rewards[4] > 0.5:
        reward = -0.1
    else:
        # if action == 1 or action == 2:
        #     reward = rewards[2] - 1.0
        reward = rewards[2] / 100.0
    print ': {:7.4f}'.format(reward)
    return reward

    # opp_r = -1.0 * (1 - rewards[3])
    # ped_r = -1.0 * rewards[4]
    # vec_r = rewards[5] / 10.0
    # combi_r = opp_r + ped_r + vec_r
    # return combi_r


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


tf.app.flags.DEFINE_string("logdir",
                           "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/"
                           "resnet_frame_skip_scale_reward_ac_v2",
                           """save tmp model""")
tf.app.flags.DEFINE_string("savedir",
                           "/home/pirate03/hobotrl_data/playground/initialD/exp/"
                           "docker005_no_stopping_static_middle_no_path_all_green/"
                           "resnet_frame_skip_scale_reward_ac_v2_records",
                           """records data""")
tf.app.flags.DEFINE_string("readme", "learn q with frame skipping. Shorten step_delay_target."
                                     "Scale return."
                                     "Turn learning on."
                                     "use q loss to instead op_loss."
                                     "Stop gradient on pi layer and conv layer"
                                     "InitialD waits until 40s."
                                     "Use new reward function.", """readme""")
tf.app.flags.DEFINE_string("port", '7024', "Docker port")
tf.app.flags.DEFINE_float("gpu_fraction", 0.6, """gpu fraction""")
tf.app.flags.DEFINE_float("discount_factor", 0.99, """actor critic discount factor""")
tf.app.flags.DEFINE_integer("batch_size", 64, """actor critic discount factor""")
tf.app.flags.DEFINE_float("lr", 0.01, """actor critic learning rate""")
tf.app.flags.DEFINE_bool("is_learn_q", False, """learn q or not""")
tf.app.flags.DEFINE_bool("is_fine_tune", True, """Stop gradient on conv layer if fine tune""")
# tf.app.flags.DEFINE_bool("use_pretrained_q", False, """learn q function or directly actor critic""")
tf.app.flags.DEFINE_bool("is_dummy_action", False, "record rule based scenes")
tf.app.flags.DEFINE_bool("learning_off", False, "learning on or off")
tf.app.flags.DEFINE_float("step_delay_target", 0.4, "learning on or off")
FLAGS = tf.app.flags.FLAGS


env = DrivingSimulatorEnv(
    address="10.31.40.197", port=FLAGS.port,
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
    step_delay_target=FLAGS.step_delay_target,
    is_dummy_action=FLAGS.is_dummy_action)


# TODO: define these Gym related params insode DrivingSimulatorEnv
stack_num = 3
env.observation_space = Box(low=0, high=255, shape=(256, 256, 9))
env.reward_range = (-np.inf, np.inf)
env.metadata = {}
env.action_space = Discrete(len(ACTIONS))
env = FrameStack(env, 3)

n_interactive = 0
n_skip = 3
n_ep = 0  # last ep in the last run, if restart use 0
state_shape = (256, 256, 9)

global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

agent = hrl.ActorCritic(
            f_create_net=f_net,
            state_shape=state_shape,
            # ACUpdate arguments
            discount_factor=FLAGS.discount_factor,
            entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-4),
            target_estimator=None,
            max_advantage=100.0,
            # optimizer arguments
            network_optmizer=hrl.network.LocalOptimizer(tf.train.AdamOptimizer(FLAGS.lr), grad_clip=10.0),
            max_gradient=10.0,
            # sampler arguments
            sampler=None,
            batch_size=FLAGS.batch_size,
            global_step=global_step,
            is_learn_q=FLAGS.is_learn_q
        )

config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False)

os.mkdir(FLAGS.savedir)
restore_var_list = []
if not FLAGS.is_learn_q:
    for var in tf.global_variables():
        print "var_name: ", var.name
        if 'Adam' in var.name or 'optimizers/beta1_power' in var.name \
                or 'optimizers/beta2_power' in var.name\
                or var.name == 'global_step:0':
            pass
        else:
            restore_var_list.append(var)
else:
    for var in tf.global_variables():
        print "var_name: ", var.name
        if 'Adam' in var.name or 'optimizers/beta1_power' in var.name \
                or 'optimizers/beta2_power' in var.name\
                or 'q_logits' in var.name\
                or var.name == 'global_step:0':
            pass
        else:
            restore_var_list.append(var)


try:
    with agent.create_session(config=config, save_dir=FLAGS.logdir, save_checkpoint_secs=1200,
                              restore_var_list=restore_var_list) as sess:
        summary_writer = SummaryWriterCache.get(FLAGS.logdir)
        all_vars = tf.global_variables()
        with open(FLAGS.logdir+"/readme.txt", "w") as f:
            f.write("readme: {}\n".format(FLAGS.readme))
            f.write("logdir: {}\n".format(FLAGS.logdir))
            f.write("savedir: {}\n".format(FLAGS.savedir))
            f.write("restore var names: \n")
            for var_name in restore_var_list:
                f.write("{}\n".format(var_name))
            f.write("gpu_fraction: {}\n".format(FLAGS.gpu_fraction))
            f.write("discount_factor: {}\n".format(FLAGS.discount_factor))
            f.write("batch_size: {}\n".format(FLAGS.batch_size))
            f.write("ac learning rate: {}\n".format(FLAGS.lr))
            f.write("is_learn_q: {} \n".format(FLAGS.is_learn_q))
            f.write("is_fine_tune: {} \n".format(FLAGS.is_fine_tune))

            # f.write("use pretrained q net: {}\n".format(FLAGS.use_pretrained_q))
            f.write("learn off: {}\n".format(FLAGS.learning_off))
            f.write("step_delay_target: {}\n".format(FLAGS.step_delay_target))
            f.write("vars: \n")
            for var in all_vars:
                f.write("{}\n".format(var.name))
                var_value = sess.run(var)
                f.write("{}\n\n".format(var_value))

        total_steps = 0
        while True:
            n_ep += 1
            n_steps = 0
            unscaled_rewards = []
            eps_dir = FLAGS.savedir + "/" + str(n_ep).zfill(4)
            os.mkdir(eps_dir)
            recording_filename = eps_dir + "/" + "0000.txt"
            recording_file = open(recording_filename, 'w')
            state = env.reset()
            state = resize_state(np.array(state))
            skip_state = state
            skip_reward = 0.0

            distribution = agent._policy._distribution.dist_run(np.asarray(state)[np.newaxis, :])
            sample_i = []
            for p in distribution:
                sample = np.random.choice(np.arange(3), p=p)
                sample_i.append(sample)
                # sample_i.append(np.argmax(p))
            sample_i = np.asarray(sample_i)
            action = sample_i[0]
            skip_action = action

            str_p = ""
            for tp in p.tolist():
                str_p += str(tp)
                str_p += ","
            str_p += "\n"

            print "========reset======\n" * 5
            while True:
                # t1 = time.time()
                # print "start time: ", t1
                next_state, vec_reward, done, _ = env.step(action)
                # t2 = time.time()
                # print "step time: ", t2 - t1
                next_state = resize_state(np.array(next_state))
                reward = func_compile_reward_agent(vec_reward)
                unscaled_rewards.append(reward * 10.0)
                skip_reward += reward
                img = state[:, :, 6:]
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
                recording_file.write(str_p)
                recording_file.write(str_reward)
                recording_file.write("\n")
                n_steps += 1
                total_steps += 1
                state = next_state

                if n_steps % n_skip == 0 or done:
                    skip_num = n_skip if n_steps % n_skip == 0 else n_steps % n_skip
                    # l1 = time.time()
                    info = agent.step(state=skip_state, action=skip_action, reward=skip_reward, next_state=state,
                                      episode_done=done, learning_off=FLAGS.learning_off)
                    # l2 = time.time()
                    # print "agent learn time: ", l2-l1
                    record(summary_writer, total_steps, info)

                    distribution = agent._policy._distribution.dist_run(np.asarray(state)[np.newaxis, :])
                    sample_i = []
                    for p in distribution:
                        sample = np.random.choice(np.arange(3), p=p)
                        sample_i.append(sample)
                        # sample_i.append(np.argmax(p))
                    sample_i = np.asarray(sample_i)
                    action = sample_i[0]

                    str_p = ""
                    for tp in p.tolist():
                        str_p += str(tp)
                        str_p += ","
                    str_p += "\n"

                    skip_state = state
                    skip_reward = 0.0
                    skip_action = action
                else:
                    action = 3
                    str_p = "0.0, 0.0, 0.0 \n"


                # t3 = time.time()
                # print "loop time: ", t3 - t1

                if done is True:
                    print "========Run Done=======\n"*5
                    break

            total_reward = 0.0
            for r in unscaled_rewards[::-1]:
                total_reward = FLAGS.discount_factor * total_reward + r
            summary = tf.Summary()
            summary.value.add(tag="episode_total_reward", simple_value=total_reward)
            summary_writer.add_summary(summary, n_ep)
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

