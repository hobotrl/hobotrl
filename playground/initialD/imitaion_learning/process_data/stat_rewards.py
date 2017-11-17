import numpy as np
import os
import math

def no_nan(rewards):
    for r in rewards:
        if np.isnan(r):
            return False
    return True


def eps_rewards(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    stat_rewards = []
    scarlar_rewards = []
    for i, line in enumerate(lines):
        # if i > len(lines):
        #     break
        if i % 2 == 1:
            vec_rewards = map(float, line.split(',')[:-1])
            stat_rewards.append(vec_rewards)
        else:
            scarlar_rewards.append(float(line.split("\n")[0].split(",")[-1]))

    for i in range(len(scarlar_rewards)):
        stat_rewards[i].append(scarlar_rewards[i])

    return filter(no_nan, stat_rewards)


def obj_rewards(obj_dir, num_eps):
    eps_names = sorted(os.listdir(obj_dir))
    vec_rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        vc = eps_rewards(eps_dir)
        vec_rewards.append(vc)
        i += 1
        if i >= num_eps:
            break
    return vec_rewards


def new_func(rewards):
    if rewards[3] < 0.5 or rewards[4] > 0.5:
        reward = -1.0
    else:
        reward = rewards[2] / 10.0
    return reward


def eps_new_func_reward(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    stat_rewards = []
    scarlar_rewards = []
    for i, line in enumerate(lines):
        if i % 2 == 1:
            vec_rewards = map(float, line.split(',')[:-1])
            stat_rewards.append(vec_rewards)
            scarlar_rewards.append(new_func(vec_rewards))
    for i in range(len(scarlar_rewards)):
        stat_rewards[i].append(scarlar_rewards[i])

    return filter(no_nan, stat_rewards)


def obj_new_func_reward(obj_dir, num_eps):
    eps_names = sorted(os.listdir(obj_dir))
    rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" +eps_name
        rewards.append(eps_new_func_reward(eps_dir))
        i += 1
        if i >= num_eps:
            break
    return rewards


def eps_rewards_v2(eps_dir):
    f = open(eps_dir + "/" + "0000.txt", "r")
    lines = f.readlines()
    stat_rewards = []
    scarlar_rewards = []
    for i, line in enumerate(lines):
        if i % 4 == 2:
            vec_rewards = map(float, line.split(',')[:-1])
            stat_rewards.append(vec_rewards)
            scarlar_rewards.append(new_func(vec_rewards))
        else:
            pass

    for i in range(len(scarlar_rewards)):
        stat_rewards[i].append(scarlar_rewards[i])

    return filter(no_nan, stat_rewards)

def obj_rewards_v2(obj_dir, num_eps=150):
    eps_names = sorted(os.listdir(obj_dir))
    obj_rewards = []
    i = 0
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        eps_rewards = eps_rewards_v2(eps_dir)
        obj_rewards.append(eps_rewards)
        i += 1
        if i >= num_eps:
            break
    return obj_rewards


def obj_ave_eps_rewards_v2(obj_dir, num_eps=300):
    vec_rewards = obj_rewards_v2(obj_dir, num_eps)
    obj_ave_rewards = []
    for eps_vec_r in vec_rewards:
        obj_ave_rewards.append(np.mean(eps_vec_r, axis=0))
    return np.mean(obj_ave_rewards, axis=0)


def disc_eps_reward(eps_vec_rewards, disc=0.99):
    disc_reward = 0.0
    eps_vec_rewards = np.array(eps_vec_rewards)
    for r in eps_vec_rewards[::-1]:
        disc_reward = r + disc * disc_reward
    return disc_reward


def obj_disc_eps_rewards(obj_dir, num_eps=300, disc=0.99, is_v2=True):
    if is_v2:
        obj_vec_rewards = obj_rewards_v2(obj_dir, num_eps)
    else:
        obj_vec_rewards = obj_new_func_reward(obj_dir, num_eps)
    obj_disc_rewards = []
    for eps_vec_rewards in obj_vec_rewards:
        obj_disc_rewards.append(disc_eps_reward(eps_vec_rewards, disc))
    return obj_disc_rewards


def stat_obj_disc_eps_rewards(obj_dir, num=300, disc=0.99, is_v2=True):
    reward = obj_disc_eps_rewards(obj_dir, num, disc, is_v2)
    for i in range(int(math.ceil(num / 100.0))):
        reward_between = reward[i*100:(i+1)*100]
        reward_to = reward[:(i+1)*100]
        print "{}-{}: {}".format(i*100, min((i+1)*100, num), np.mean(reward_between, axis=0))
        print "{}: {} \n".format(min((i+1)*100, num), np.mean(reward_to, axis=0))


def stat_obj_disc_eps_rewards_v2_list(obj_dir_list, num_list, disc_list, is_v2_list):
    for obj_dir, num, disc, is_v2 in zip(obj_dir_list, num_list, disc_list, is_v2_list):
        print obj_dir, "\n"
        stat_obj_disc_eps_rewards(obj_dir, num, disc, is_v2)



def stat(obj_dir, num, is_v2=True):
    print obj_dir
    if is_v2:
        reward = obj_rewards_v2(obj_dir, num)
    else:
        reward = obj_new_func_reward(obj_dir, num)
    for i in range(int(math.ceil(num / 100.0))):
        reward_between = [v for vs in reward[i*100:(i+1)*100] for v in vs]
        reward_to = [v for vs in reward[:(i+1)*100] for v in vs]
        print "{}-{}: {}".format(i*100, min((i+1)*100, num), np.mean(reward_between, axis=0))
        print "{}: {} \n".format(min((i+1)*100, num), np.mean(reward_to, axis=0))


def stat_list(obj_dir_list, num_list, is_v2_list):
    for obj_dir, num, is_v2 in zip(obj_dir_list, num_list, is_v2_list):
        stat(obj_dir, num, is_v2)


def stat_eps_ave(obj_dir, num, is_v2=True):
    print obj_dir
    if is_v2:
        reward = obj_rewards_v2(obj_dir, num)
    else:
        reward = obj_new_func_reward(obj_dir, num)
    for i in range(int(math.ceil(num / 100.0))):
        reward_between = [np.mean(vs, axis=0) for vs in reward[i*100:(i+1)*100]]
        reward_to = [np.mean(vs, axis=0) for vs in reward[:(i+1)*100]]
        print "{}-{}: {}".format(i*100, min((i+1)*100, num), np.mean(reward_between, axis=0))
        print "{}: {} \n".format(min((i+1)*100, num), np.mean(reward_to, axis=0))

def stat_eps_ave_list(obj_dir_list, num_list, is_v2_list):
    for obj_dir, num, is_v2 in zip(obj_dir_list, num_list, is_v2_list):
        stat_eps_ave(obj_dir, num, is_v2)



import matplotlib.pyplot as plt

def plot_box(obj_dir_list, num_list, is_v2_list, title_list=''):
    print obj_dir_list

    plt.title("reward comparision")
    for i in range(len(obj_dir_list)):
        if is_v2_list[i]:
            obj_vec_rewards = obj_rewards_v2(obj_dir, num)
        else:
            obj_vec_rewards = obj_rewards(obj_dir, num)

        obj_mean_rewards = []
        for eps_vec_rewards in obj_vec_rewards:
            obj_mean_rewards.append(np.mean(eps_vec_rewards, axis=0))

    obj_mean_rewards = np.array(obj_mean_rewards)
    plt.plot(range(1, num+1), obj_mean_rewards[:, 2], '.')
    plt.ylabel('reward')
    plt.show()



if __name__ == "__main__":

    # obj80_not_stop_gradient_imitation_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_adam0001_not_stop_gradient_records"
    # obj80_not_stop_gradient_imitation_reward = obj_rewards(obj80_not_stop_gradient_imitation_dir, 300)
    # obj80_not_stop_gradient_imitation_reward_200 = [v for vs in obj80_not_stop_gradient_imitation_reward[:200] for v in vs]
    # obj80_not_stop_gradient_imitation_reward_250 = [v for vs in obj80_not_stop_gradient_imitation_reward[:250] for v in vs]
    # obj80_not_stop_gradient_imitation_reward_300 = [v for vs in obj80_not_stop_gradient_imitation_reward[:300] for v in vs]
    # obj80_not_stop_gradient_imitation_reward_100 = [v for vs in obj80_not_stop_gradient_imitation_reward[:100] for v in vs]
    # obj80_not_stop_gradient_imitation_reward_100_200 = [v for vs in obj80_not_stop_gradient_imitation_reward[100:200] for v in vs]
    # obj80_not_stop_gradient_imitation_reward_200_300 = [v for vs in obj80_not_stop_gradient_imitation_reward[200:300] for v in vs]
    # print "obj80_not_stop_gradient_imitation_include_red_line_reward: "
    # print "200: ", np.mean(obj80_not_stop_gradient_imitation_reward_200, axis=0), np.var(obj80_not_stop_gradient_imitation_reward_200, axis=0)
    # print "250: ", np.mean(obj80_not_stop_gradient_imitation_reward_250, axis=0), np.var(obj80_not_stop_gradient_imitation_reward_250, axis=0)
    # print "300: ", np.mean(obj80_not_stop_gradient_imitation_reward_300, axis=0), np.var(obj80_not_stop_gradient_imitation_reward_300, axis=0)
    # print "100: ", np.mean(obj80_not_stop_gradient_imitation_reward_100, axis=0)
    # print "100-200: ", np.mean(obj80_not_stop_gradient_imitation_reward_100_200, axis=0)
    # print "200-300: ", np.mean(obj80_not_stop_gradient_imitation_reward_200_300, axis=0)
    # print "\n"
    #
    # obj80_stop_gradient_imitation_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_obj80_adam001_stop_gradient_v0_records"
    # obj80_stop_gradient_imitation_reward = obj_rewards_v2(obj80_stop_gradient_imitation_dir, 300)
    # obj80_stop_gradient_imitation_reward_200 = [v for vs in obj80_stop_gradient_imitation_reward[:200] for v in vs]
    # obj80_stop_gradient_imitation_reward_250 = [v for vs in obj80_stop_gradient_imitation_reward[:250] for v in vs]
    # obj80_stop_gradient_imitation_reward_300 = [v for vs in obj80_stop_gradient_imitation_reward[:300] for v in vs]
    # obj80_stop_gradient_imitation_reward_100 = [v for vs in obj80_stop_gradient_imitation_reward[:100] for v in vs]
    # obj80_stop_gradient_imitation_reward_100_200 = [v for vs in obj80_stop_gradient_imitation_reward[100:200] for v in vs]
    # obj80_stop_gradient_imitation_reward_200_300 = [v for vs in obj80_stop_gradient_imitation_reward[200:300] for v in vs]
    # print "obj80_stop_gradient_imitation_exclude_red_line_reward: "
    # print "200: ", np.mean(obj80_stop_gradient_imitation_reward_200, axis=0), np.var(obj80_stop_gradient_imitation_reward_200, axis=0)
    # print "250: ", np.mean(obj80_stop_gradient_imitation_reward_250, axis=0), np.var(obj80_stop_gradient_imitation_reward_250, axis=0)
    # print "300: ", np.mean(obj80_stop_gradient_imitation_reward_300, axis=0), np.var(obj80_stop_gradient_imitation_reward_300, axis=0)
    # print "100: ", np.mean(obj80_stop_gradient_imitation_reward_100, axis=0)
    # print "100-200: ", np.mean(obj80_stop_gradient_imitation_reward_100_200, axis=0)
    # print "200-300: ", np.mean(obj80_stop_gradient_imitation_reward_200_300, axis=0)
    # print "\n"
    #
    # obj80_rule_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_docker005_all_green_obj80"
    # obj80_rule_reward = obj_rewards(obj80_rule_dir, 300)
    # obj80_rule_reward_200 = [v for vs in obj80_rule_reward[:200] for v in vs]
    # obj80_rule_reward_250 = [v for vs in obj80_rule_reward[:250] for v in vs]
    # obj80_rule_reward_300 = [v for vs in obj80_rule_reward[:300] for v in vs]
    # obj80_rule_reward_100 = [v for vs in obj80_rule_reward[:100] for v in vs]
    # obj80_rule_reward_100_200 = [v for vs in obj80_rule_reward[100:200] for v in vs]
    # obj80_rule_reward_200_300 = [v for vs in obj80_rule_reward[200:300] for v in vs]
    # print "obj80_rule_exclude_red_line_reward: "
    # print "200: ", np.mean(obj80_rule_reward_200, axis=0), np.var(obj80_rule_reward_200, axis=0)
    # print "250: ", np.mean(obj80_rule_reward_250, axis=0), np.var(obj80_rule_reward_250, axis=0)
    # print "300: ", np.mean(obj80_rule_reward_300, axis=0), np.var(obj80_rule_reward_300, axis=0)
    # print "100: ", np.mean(obj80_rule_reward_100, axis=0), np.var(obj80_rule_reward_100, axis=0)
    # print "100-200: ", np.mean(obj80_rule_reward_100_200, axis=0), np.var(obj80_rule_reward_100_200, axis=0)
    # print "200-300: ", np.mean(obj80_rule_reward_200_300, axis=0), np.var(obj80_rule_reward_200_300, axis=0)
    # print "\n"
    #
    # obj100_rule_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/" \
    #                   "rule_scenes_vec_rewards_maybe_obj_100_docker005_no_early_stopping_all_green_orig_back_records/" \
    #                   "record_rule_scenes_vec_rewards_obj80_docker005_no_early_stopping_all_green"
    # obj100_rule_reward = obj_rewards(obj100_rule_dir, 500)
    # obj100_rule_reward_200 = [v for vs in obj100_rule_reward[:200] for v in vs]
    # obj100_rule_reward_250 = [v for vs in obj100_rule_reward[:250] for v in vs]
    # obj100_rule_reward_300 = [v for vs in obj100_rule_reward[:300] for v in vs]
    # obj100_rule_reward_400 = [v for vs in obj100_rule_reward[:400] for v in vs]
    # obj100_rule_reward_500 = [v for vs in obj100_rule_reward[:500] for v in vs]
    # print "obj100_rule_reward: "
    # print "200: ", np.mean(obj100_rule_reward_200, axis=0), np.var(obj100_rule_reward_200, axis=0)
    # print "250: ", np.mean(obj100_rule_reward_250, axis=0), np.var(obj100_rule_reward_250, axis=0)
    # print "300: ", np.mean(obj100_rule_reward_300, axis=0), np.var(obj100_rule_reward_300, axis=0)
    # print "400: ", np.mean(obj100_rule_reward_400, axis=0), np.var(obj100_rule_reward_400, axis=0)
    # print "500: ", np.mean(obj100_rule_reward_500, axis=0), np.var(obj100_rule_reward_500, axis=0)
    # print "\n"


    # obj100_train_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/train"
    # obj100_train_reward = obj_rewards(obj100_train_dir, 300)
    # obj100_train_reward_200 = [v for vs in obj100_train_reward[:200] for v in vs]
    # obj100_train_reward_250 = [v for vs in obj100_train_reward[:250] for v in vs]
    # obj100_train_reward_300 = [v for vs in obj100_train_reward[:300] for v in vs]
    # print "obj100_train_reward: "
    # print "200: ", np.mean(obj100_train_reward_200, axis=0), np.var(obj100_train_reward_200, axis=0)
    # print "250: ", np.mean(obj100_train_reward_250, axis=0), np.var(obj100_train_reward_250, axis=0)
    # print "300: ", np.mean(obj100_train_reward_300, axis=0), np.var(obj100_train_reward_300, axis=0)


    # obj80_ac_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_ac_records"
    # obj80_ac_reward = obj_rewards_v2(obj80_ac_dir, 1200)
    # obj80_ac_reward_100 = [v for vs in obj80_ac_reward[:100] for v in vs]
    # obj80_ac_reward_100_200 = [v for vs in obj80_ac_reward[100:200] for v in vs]
    # obj80_ac_reward_200_300 = [v for vs in obj80_ac_reward[200:300] for v in vs]
    # obj80_ac_reward_200_350 = [v for vs in obj80_ac_reward[200:350] for v in vs]
    # obj80_ac_reward_350_450 = [v for vs in obj80_ac_reward[350:450] for v in vs]
    # obj80_ac_reward_500_600 = [v for vs in obj80_ac_reward[500:600] for v in vs]
    # # obj80_ac_reward_600_650 = [v for vs in obj80_ac_reward[600:650] for v in vs]
    # obj80_ac_reward_700_800 = [v for vs in obj80_ac_reward[700:800] for v in vs]
    # obj80_ac_reward_800_900 = [v for vs in obj80_ac_reward[800:900] for v in vs]
    # obj80_ac_reward_900_980 = [v for vs in obj80_ac_reward[900:980] for v in vs]
    # obj80_ac_reward_1100_1200 = [v for vs in obj80_ac_reward[1100:1200] for v in vs]
    # print "obj80_ac_reward: "
    # print "100: ", np.mean(obj80_ac_reward_100, axis=0), np.var(obj80_ac_reward_100, axis=0)
    # print "100-200: ", np.mean(obj80_ac_reward_100_200, axis=0), np.var(obj80_ac_reward_100_200, axis=0)
    # print "200-300: ", np.mean(obj80_ac_reward_200_300, axis=0), np.var(obj80_ac_reward_200_300, axis=0)
    # print "200-350: ", np.mean(obj80_ac_reward_200_350, axis=0), np.var(obj80_ac_reward_200_350, axis=0)
    # print "350-450: ", np.mean(obj80_ac_reward_350_450, axis=0)
    # print "500-600: ", np.mean(obj80_ac_reward_500_600, axis=0)
    # print "700-800: ", np.mean(obj80_ac_reward_700_800, axis=0)
    # print "800-900: ", np.mean(obj80_ac_reward_800_900, axis=0)
    # print "900-980: ", np.mean(obj80_ac_reward_900_980, axis=0)
    # print "1100-1200: ", np.mean(obj80_ac_reward_1100_1200, axis=0)
    # print "\n"


    # obj80_stop_gradient_learn_q_v1_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_learn_q_obj80_adam001_stop_gradient_v1_records"
    # obj80_stop_gradient_learn_q_v1_reward = obj_rewards_v2(obj80_stop_gradient_learn_q_v1_dir, 350)
    # obj80_stop_gradient_learn_q_v1_reward_100 = [v for vs in obj80_stop_gradient_learn_q_v1_reward[:100] for v in vs]
    # obj80_stop_gradient_learn_q_v1_reward_200 = [v for vs in obj80_stop_gradient_learn_q_v1_reward[:200] for v in vs]
    # obj80_stop_gradient_learn_q_v1_reward_300 = [v for vs in obj80_stop_gradient_learn_q_v1_reward[:300] for v in vs]
    # obj80_stop_gradient_learn_q_v1_reward_350 = [v for vs in obj80_stop_gradient_learn_q_v1_reward[:350] for v in vs]
    # print "obj80_stop_gradient_learn_q_v1_reward: "
    # print "100: ", np.mean(obj80_stop_gradient_learn_q_v1_reward_100, axis=0)
    # print "200: ", np.mean(obj80_stop_gradient_learn_q_v1_reward_200, axis=0)
    # print "300: ", np.mean(obj80_stop_gradient_learn_q_v1_reward_300, axis=0)
    # print "350: ", np.mean(obj80_stop_gradient_learn_q_v1_reward_350, axis=0)
    # print "\n"
    #
    #
    # obj80_rnd_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/rnd"
    # obj80_rnd_reward = obj_rewards(obj80_rnd_dir, 120)
    # obj80_rnd_reward_120 = [v for vs in obj80_rnd_reward[:120] for v in vs]
    # print "obj80_rnd_reward: "
    # print "120: ", np.mean(obj80_rnd_reward_120, axis=0)
    # print "\n"


    # obj80_ac_wait40s_new_reward_func_only_q_loss_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/" \
    #                                                    "resnet_ac_wait_40s_new_func_reward_only_q_loss"
    # obj80_ac_wait40s_new_reward_func_only_q_loss_reward = obj_rewards_v2(obj80_ac_wait40s_new_reward_func_only_q_loss_dir, 100)
    # obj80_ac_new_reward_func_only_q_loss_80 = [v for vs in obj80_ac_wait40s_new_reward_func_only_q_loss_reward[:80] for v in vs]
    # obj80_ac_new_reward_func_only_q_loss_100 = [v for vs in  obj80_ac_wait40s_new_reward_func_only_q_loss_reward[:100] for v in vs]
    # print "ac new reward func only q loss: "
    # print "80: ", np.mean(obj80_ac_new_reward_func_only_q_loss_80, axis=0)
    # print "100: ", np.mean(obj80_ac_new_reward_func_only_q_loss_100, axis=0)
    #
    # obj80_ac_wait40s_new_reward_func_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/" \
    #                                        "docker005_no_stopping_static_middle_no_path_all_green/" \
    #                                        "resnet_ac_wait_40s_new_func_reward_records"
    # obj80_ac_wait40s_new_reward_func_reward = obj_rewards_v2(obj80_ac_wait40s_new_reward_func_dir, 500)
    # obj80_ac_reward_100 = [v for vs in obj80_ac_wait40s_new_reward_func_reward[:100] for v in vs]
    # obj80_ac_reward_200 = [v for vs in obj80_ac_wait40s_new_reward_func_reward[:200] for v in vs]
    # obj80_ac_reward_300 = [v for vs in obj80_ac_wait40s_new_reward_func_reward[:300] for v in vs]
    # obj80_ac_reward_100_200 = [v for vs in obj80_ac_wait40s_new_reward_func_reward[100:200] for v in vs]
    # obj80_ac_reward_200_300 = [v for vs in obj80_ac_wait40s_new_reward_func_reward[200:300] for v in vs]
    # obj80_ac_reward_300_400 = [v for vs in obj80_ac_wait40s_new_reward_func_reward[300:400] for v in vs]
    # obj80_ac_reward_400_500 = [v for vs in obj80_ac_wait40s_new_reward_func_reward[400:500] for v in vs]
    # print "ac new reward func 100: ", np.mean(obj80_ac_reward_100, axis=0)
    # print "ac new reward func 200: ", np.mean(obj80_ac_reward_200, axis=0)
    # print "ac new reward func 300: ", np.mean(obj80_ac_reward_300, axis=0)
    # print "ac new reward func 100-200: ", np.mean(obj80_ac_reward_100_200, axis=0)
    # print "ac new reward func 200-300: ", np.mean(obj80_ac_reward_200_300, axis=0)
    # print "ac new reward func 300-400: ", np.mean(obj80_ac_reward_300_400, axis=0)
    # print "ac new reward func 400-500: ", np.mean(obj80_ac_reward_400_500, axis=0)


    obj_dir_list = [
                    # '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_check_learn_q_wait_40s_new_func_reward_records',
                    # '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_check_learn_q_wait_40s_new_func_reward_no_q_records',
                    # '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_pq_ac_stop_conv4_include_red_line_reward_records',
                    '/home/pirate03/work/agents/recording_data/resnet_check_no_q_wait_40s_new_func_reward_learning_off_records',
                    # '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/repeat_learn_q_v0_turn_learn_on_only_q_loss_records'
                    # '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/repeat_learn_q_v0_turn_learn_off'
        '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/record_rule_docker005_all_green_obj80',
        '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/wait_40s/resnet_learn_q_wait40s_records',
        '/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/wait_40s/resnet_ac_with_q_learned_from_wait40s_records',
        "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_frame_skip_scale_reward_ac_records",
        "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_frame_skip_scale_reward_test_model_records",
        "/home/pirate03/hobotrl_data/playground/initialD/exp/docker006_frame_skip/learn_q_records",
        "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_frame_skip_scale_reward_wrong_total_reward/ac_records",
        "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_frame_skip_scale_reward_wrong_total_reward/learn_q_records"
    ]
    num_list = [600, 500, 350, 1000, 650, 250, 300, 300, 900]
    is_v2_list = [True, False, True, True, True, True, True, True, True]
    print "zhankai eps: "
    stat_list(obj_dir_list, num_list, is_v2_list)

    # print "\n\n"
    # print "eps ave: "
    # stat_eps_ave_list(obj_dir_list, num_list, is_v2_list)
    # disc_list = [0.99, 0.99, 0.99, 0.99]
    # stat_obj_disc_eps_rewards_v2_list(obj_dir_list, num_list, disc_list, is_v2_list)


    # obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/docker005_no_stopping_static_middle_no_path_all_green/resnet_ac_with_q_learned_from_wait40s_records"
    # num = 1400
    # stat_obj_disc_eps_rewards_v2(obj_dir, num=num, disc=0.99)

    # plot_box(obj_dir, num)
