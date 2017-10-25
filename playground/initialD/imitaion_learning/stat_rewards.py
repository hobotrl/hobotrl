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


def eps_rewards_v2(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    scarlar_rewards = []
    stat_rewards = []
    for i, line in enumerate(lines):
        # if i > len(lines):
        #     break
        if i % 4 == 0:
            # info = float(line.split("\n")[0].split(",")[-1])
            scarlar_rewards.append(float(line.split("\n")[0].split(",")[-1]))
        elif i % 4 == 2:
            vec_rewards = map(float, line.split(',')[:-1])
            stat_rewards.append(vec_rewards)
        else:
            pass

    for i in range(len(scarlar_rewards)):
        stat_rewards[i].append(scarlar_rewards[i])

    return filter(no_nan, stat_rewards)



def obj_rewards_v2(obj_dir, num_eps=150):
    eps_names = sorted(os.listdir(obj_dir))
    vec_rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        vc = eps_rewards_v2(eps_dir)
        vec_rewards.extend(vc)
        i += 1
        if i >= num_eps:
            break
    return vec_rewards


def eps_vel(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    stat_vels = []
    for i, line in enumerate(lines):
        if i % 2 == 1:
            vel = float(line.split(',')[2])
            stat_vels.append(vel)
    return filter(lambda x: False if np.isnan(x) else True, stat_vels)


def obj_vel(obj_dir, num_eps=300):
    eps_names = sorted(os.listdir(obj_dir))
    vec_rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        vc = eps_vel(eps_dir)
        vec_rewards.append(vc)
        i += 1
        if i >= num_eps:
            break
    return vec_rewards


def eps_vel_v2(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    stat_vels = []
    for i, line in enumerate(lines):
        if i % 4 == 2:
            vel = float(line.split(',')[2])
            stat_vels.append(vel)
    return filter(lambda x: False if np.isnan(x) else True, stat_vels)


def obj_vel_v2(obj_dir, num_eps=300):
    eps_names = sorted(os.listdir(obj_dir))
    vec_rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        vc = eps_vel_v2(eps_dir)
        vec_rewards.append(vc)
        i += 1
        if i >= num_eps:
            break
    return vec_rewards


if __name__ == "__main__":

    # ac_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_ac_no_early_stopping"
    # rnd_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80_vec_rewards_all_green_docker005"
    # rule_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_docker005_all_green_obj80"
    # rule_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_obj_80_docker005_no_early_stopping_all_green_orig_back/record_rule_scenes_vec_rewards_obj80_docker005_no_early_stopping_all_green"
    # ac_vr = obj_rewards(ac_obj_dir)
    # rnd_vec_reward = obj_rewards(rnd_obj_dir, 100)
    # rule_vec_reward = obj_rewards(rule_obj_dir, 300)
    # im_vec_reward = obj_rewards(im_obj_dir, 300)
    # im_vec_reward = obj_rewards_v2(im_obj_dir, 300)
    # ac_av_vr = np.mean(ac_vr, axis=0)
    # rnd_mean = np.mean(rnd_vec_reward, axis=0)
    # rnd_var = np.var(rnd_vec_reward, axis=0)
    # rule_mean = np.mean(rule_vec_reward, axis=0)
    # rule_var = np.var(rule_vec_reward, axis=0)
    # im_mean = np.mean(im_vec_reward, axis=0)
    # im_var = np.var(im_vec_reward, axis=0)
    # print "ac: ", ac_av_vr
    # print "rnd: ", rnd_mean, ", ", rnd_var
    # print "rule: ", rule_mean, ", ", rule_var
    # print "im: ", im_mean, ", ", im_var


    obj80_not_stop_gradient_imitation_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_adam0001_not_stop_gradient_records"
    obj80_stop_gradient_imitation_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_obj80_adam001"


    obj80_not_stop_gradient_imitation_vels = obj_vel(obj80_not_stop_gradient_imitation_dir, 300)
    obj80_not_stop_gradient_imitation_vel_100 = [v for vs in obj80_not_stop_gradient_imitation_vels[:100] for v in vs]
    obj80_not_stop_gradient_imitation_vel_150 = [v for vs in obj80_not_stop_gradient_imitation_vels[:150] for v in vs]
    obj80_not_stop_gradient_imitation_vel_200 = [v for vs in obj80_not_stop_gradient_imitation_vels[:200] for v in vs]
    obj80_not_stop_gradient_imitation_vel_250 = [v for vs in obj80_not_stop_gradient_imitation_vels[:250] for v in vs]
    obj80_not_stop_gradient_imitation_vel_300 = [v for vs in obj80_not_stop_gradient_imitation_vels[:300] for v in vs]
    # with open("obj80_not_stop_gradient_imitation.txt", "w") as f:
    #     for v in obj80_not_stop_gradient_imitation_vel:
    #         f.write(str(v))
    #         f.write("\n")
    print "obj80_not_stop_gradient_imitation_include_red_line_vel: "
    print "100: ", np.mean(obj80_not_stop_gradient_imitation_vel_100), np.var(obj80_not_stop_gradient_imitation_vel_100)
    print "150: ", np.mean(obj80_not_stop_gradient_imitation_vel_150), np.var(obj80_not_stop_gradient_imitation_vel_150)
    print "200: ", np.mean(obj80_not_stop_gradient_imitation_vel_200), np.var(obj80_not_stop_gradient_imitation_vel_200)
    print "250: ", np.mean(obj80_not_stop_gradient_imitation_vel_250), np.var(obj80_not_stop_gradient_imitation_vel_250)
    print "300: ", np.mean(obj80_not_stop_gradient_imitation_vel_300), np.var(obj80_not_stop_gradient_imitation_vel_300)

    obj80_stop_gradient_imitation_vels = obj_vel_v2(obj80_stop_gradient_imitation_dir, 300)
    obj80_stop_gradient_imitation_vel_100 = [v for vs in obj80_stop_gradient_imitation_vels[:100] for v in vs]
    obj80_stop_gradient_imitation_vel_150 = [v for vs in obj80_stop_gradient_imitation_vels[:150] for v in vs]
    obj80_stop_gradient_imitation_vel_200 = [v for vs in obj80_stop_gradient_imitation_vels[:200] for v in vs]
    obj80_stop_gradient_imitation_vel_250 = [v for vs in obj80_stop_gradient_imitation_vels[:250] for v in vs]
    obj80_stop_gradient_imitation_vel_300 = [v for vs in obj80_stop_gradient_imitation_vels[:300] for v in vs]
    print "obj80_stop_gradient_imitation_exclude_red_line_vel: "
    print "100: ", np.mean(obj80_stop_gradient_imitation_vel_100), np.var(obj80_stop_gradient_imitation_vel_100)
    print "150: ", np.mean(obj80_stop_gradient_imitation_vel_150), np.var(obj80_stop_gradient_imitation_vel_150)
    print "200: ", np.mean(obj80_stop_gradient_imitation_vel_200), np.var(obj80_stop_gradient_imitation_vel_200)
    print "250: ", np.mean(obj80_stop_gradient_imitation_vel_250), np.var(obj80_stop_gradient_imitation_vel_250)
    print "300: ", np.mean(obj80_stop_gradient_imitation_vel_300), np.var(obj80_stop_gradient_imitation_vel_300)

    obj80_rule_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_docker005_all_green_obj80"
    obj80_rule_vels = obj_vel(obj80_rule_dir, 300)
    obj80_rule_vel_100 = [v for vs in obj80_rule_vels[:100] for v in vs]
    obj80_rule_vel_150 = [v for vs in obj80_rule_vels[:150] for v in vs]
    obj80_rule_vel_200 = [v for vs in obj80_rule_vels[:200] for v in vs]
    obj80_rule_vel_250 = [v for vs in obj80_rule_vels[:250] for v in vs]
    obj80_rule_vel_300 = [v for vs in obj80_rule_vels[:300] for v in vs]
    print "obj80_rule_exclude_red_line_vel: "
    print "100: ", np.mean(obj80_rule_vel_100), np.var(obj80_rule_vel_100)
    print "150: ", np.mean(obj80_rule_vel_150), np.var(obj80_rule_vel_150)
    print "200: ", np.mean(obj80_rule_vel_200), np.var(obj80_rule_vel_200)
    print "250: ", np.mean(obj80_rule_vel_250), np.var(obj80_rule_vel_250)
    print "300: ", np.mean(obj80_rule_vel_300), np.var(obj80_rule_vel_300)






