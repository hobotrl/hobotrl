import numpy as np
import os


def eps_rewards(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    rewards = []
    for i, line in enumerate(lines):
        # if i > len(lines):
        #     break
        rewards.append(float(line.split("\n")[0].split(",")[-1]))
    # return rewards[:-30]
    return rewards


def obj_rewards(obj_dir, num_eps=300):
    eps_names = sorted(os.listdir(obj_dir))
    rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        rewards.extend(eps_rewards(eps_dir))
        i += 1
        if i >= num_eps:
            break
    return rewards


if __name__ == "__main__":
    rule_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai"
    # im_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q_wrong"
    im_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q_no_skip"
    rnd_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80"

    total_reward1 = obj_rewards(rule_obj_dir)
    print total_reward1
    for i, r in enumerate(total_reward1):
        if np.isnan(r):
            print i
            total_reward1[i] = 0.0

    ave_reward1 = np.mean(total_reward1)
    # max_reward1 = np.max(total_reward1)
    # print max_reward1
    total_reward2 = obj_rewards(im_obj_dir)
    ave_reward2 = np.mean(total_reward2)
    total_reward3 = obj_rewards(rnd_obj_dir)
    ave_reward3 = np.mean(total_reward3)
    print "rule ave reward: ", ave_reward1
    print "im ave reward: ", ave_reward2
    print "rnd ave reward: ", ave_reward3


    # frame_skip_im_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q"
    # total_reward4 = obj_rewards(frame_skip_im_obj_dir, num_eps=40)
    # print "im ave reward: ", np.mean(total_reward4)

