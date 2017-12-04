import os
import tensorflow as tf
import numpy as np
import cv2

def print_qvals(n_steps, agent, state, action, AGENT_ACTIONS):
    q_vals = agent.learn_q(np.asarray(state)[np.newaxis, :])[0]
    # ('s', q), ('d', q), ('a', q)
    p_dict = sorted(zip(
        map(lambda x: x[0], AGENT_ACTIONS), q_vals
    ))
    max_idx = np.argmax([v for _, v in p_dict])
    p_str = "(({:3d})[Q_vals]: ".format(n_steps)
    for i, (a, v) in enumerate(p_dict):
        if a == AGENT_ACTIONS[action][0]:
            sym = '|x|' if i == max_idx else ' x '
        else:
            sym = '| |' if i == max_idx else '   '
        p_str += '{}{:3d}: {:8.4f} '.format(sym, a, v)
    print p_str

def log_info(agent_info, env_info,
             done,
             cum_reward,
             n_ep, n_ep_steps, n_total_steps):
    summary_proto = tf.Summary()
    for tag in agent_info:
        summary_proto.value.add(
            tag=tag, simple_value=np.mean(agent_info[tag])
        )
    if done:
        summary_proto.value.add(tag='exp/n_ep_steps', simple_value=n_ep_steps)
        summary_proto.value.add(tag='exp/n_total_steps',
                                simple_value=n_total_steps)
        summary_proto.value.add(
            tag='exp/num_episodes', simple_value=n_ep)
        summary_proto.value.add(
            tag='exp/total_reward', simple_value=cum_reward)
        summary_proto.value.add(
            tag='exp/per_step_reward', simple_value=cum_reward/n_ep_steps)
        if 'flag_success' in env_info:
            summary_proto.value.add(
                tag='exp/flag_success', simple_value=env_info['flag_success'])

    return summary_proto

class StepsSaver(object):
    def __init__(self, savedir):
        self.savedir = savedir
        os.makedirs(self.savedir)
        self.eps_dir = None
        self.file = None
        self.stat_file = open(self.savedir + "/0000.txt", "a", 0)

    def close(self):
        self.stat_file.close()

    def parse_state(self, state):
        return np.array(state)[:, :, 6:]

    def save(self, n_ep, n_step, state, action, vec_reward, reward,
                  done, cum_reward, flag_success):
        if self.file is None:
            self.eps_dir = self.savedir + "/" + str(n_ep).zfill(4)
            os.mkdir(self.eps_dir)
            self.file = open(self.eps_dir + "/0000.txt", "w")

        img_path = self.eps_dir + "/" + str(n_step + 1).zfill(4) + "_" + str(action) + ".jpg"
        cv2.imwrite(
            img_path,
            cv2.cvtColor(self.parse_state(state), cv2.COLOR_RGB2BGR)
        )
        self.file.write(str(n_step) + ',' + str(action) + ',' + str(reward) + '\n')
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