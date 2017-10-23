import numpy as np
import os

def no_nan(rewards):
    for r in rewards:
        if np.isnan(r):
            return False
    return True


def eps_rewards(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    scarlar_rewards = []
    stat_rewards = []
    for i, line in enumerate(lines):
        # if i > len(lines):
        #     break
        if i % 2 == 0:
            # info = float(line.split("\n")[0].split(",")[-1])
            scarlar_rewards.append(float(line.split("\n")[0].split(",")[-1]))
        else:
            vec_rewards = map(float, line.split(',')[:-1])
            stat_rewards.append(vec_rewards)

    for i in range(len(scarlar_rewards)):
        stat_rewards[i].append(scarlar_rewards[i])

    return filter(no_nan, stat_rewards)


def obj_rewards(obj_dir, num_eps=150):
    eps_names = sorted(os.listdir(obj_dir))
    vec_rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        vc = eps_rewards(eps_dir)
        vec_rewards.extend(vc)
        i += 1
        if i >= num_eps:
            break
    return vec_rewards





if __name__ == "__main__":
    # rule_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai"
    # # im_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q_wrong"
    # im_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q_no_skip"
    # rnd_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80"
    #
    # total_reward1 = obj_rewards(rule_obj_dir)
    # print total_reward1
    # for i, r in enumerate(total_reward1):
    #     if np.isnan(r):
    #         print i
    #         total_reward1[i] = 0.0
    #
    # ave_reward1 = np.mean(total_reward1)
    # # max_reward1 = np.max(total_reward1)
    # # print max_reward1
    # total_reward2 = obj_rewards(im_obj_dir)
    # ave_reward2 = np.mean(total_reward2)
    # total_reward3 = obj_rewards(rnd_obj_dir)
    # ave_reward3 = np.mean(total_reward3)
    # print "rule ave reward: ", ave_reward1
    # print "im ave reward: ", ave_reward2
    # print "rnd ave reward: ", ave_reward3

    # frame_skip_im_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q"
    # total_reward4 = obj_rewards(frame_skip_im_obj_dir, num_eps=40)
    # print "im ave reward: ", np.mean(total_reward4)

    # im_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/fnet_rename_learn_q_skip2_vec_reward"
    # rnd_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80_vec_rewards"
    # rule_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards"

    # ac_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_ac_no_early_stopping"
    rnd_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80_docker005_no_early_stopping"
    rule_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_obj80_docker005_no_early_stopping"
    # ac_vr = obj_rewards(ac_obj_dir)
    rnd_vr = obj_rewards(rnd_obj_dir)
    rule_vr = obj_rewards(rule_obj_dir)

    # ac_av_vr = np.mean(ac_vr, axis=0)
    rnd_av_vr = np.mean(rnd_vr, axis=0)
    rule_av_vr = np.mean(rule_vr, axis=0)

    # print "ac: ", ac_av_vr
    print "rnd: ", rnd_av_vr
    print "rule: ", rule_av_vr