import os
import numpy as np

def stat_eps_action_num(eps_dir):
    filenames = sorted(os.listdir(eps_dir))
    img_names = filenames[1:]
    stat_acts = np.zeros(9)
    for img_name in img_names:
        act = int(img_name.split('.')[0].split('_')[-1])
        stat_acts[act] += 1
    return stat_acts


def stat_obj_action_num(obj_dir):
    eps_names = sorted(os.listdir(obj_dir))
    obj_stats = np.zeros(9)
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        eps_stats = stat_eps_action_num(eps_dir)
        obj_stats += eps_stats
    return obj_stats


def stat_action_start_time(eps_dir):
    file_names = sorted(os.listdir(eps_dir))
    img_names = file_names[1:]
    acts = []
    for img_name in img_names:
        act = int(img_name.split('.')[0].split('_')[-1])
        acts.append(act)

    if acts[0] != 0:
        return -1
    for i in range(10):
        if acts[i] != 0:
            break
    return i+1

def stat_action_obj_start_time(obj_dir):
    eps_names = sorted(os.listdir(obj_dir))
    stat_names = []
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        ind = stat_action_start_time(eps_dir)
        if ind >= 1 and ind < 10:
            print eps_name, ": ", ind
            stat_names.append(eps_name)
    print len(stat_names)


def rm_start_time_imgs(eps_dir, start_time, stack_num=3):
    assert start_time > stack_num
    img_names = sorted(os.listdir(eps_dir))[1:]
    for i in range(start_time-stack_num-1):
        img_path = eps_dir + "/" + img_names[i]
        os.remove(img_path)


def rm_obj_start_time_imgs(obj_dir, stack_num=3):
    eps_names = sorted(os.listdir(obj_dir))
    stat_names = []
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        ind = stat_action_start_time(eps_dir)
        if ind > stack_num and ind < 10:
            print eps_name, ": ", ind
            rm_start_time_imgs(eps_dir, ind, stack_num)
    # print len(stat_names)


def rm_eps_pred_acts(eps_dir):
    img_names = sorted(os.listdir(eps_dir))[1:]
    for img_name in img_names[:-3]:
        new_img_name = img_name[:-2]
        os.rename(eps_dir+"/"+img_name, eps_dir+"/"+new_img_name)

def rm_obj_pred_acts(obj_dir):
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        rm_eps_pred_acts(obj_dir+"/"+eps_name)

def complete_last_three_imgs(obj_dir):
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        last_three_img_names = sorted(os.listdir(eps_dir))[-3:]
        for name in last_three_img_names:
            os.rename(eps_dir+"/"+name, eps_dir+"/"+name+"pg")


def rm_obj_txt_start_time_lines(obj_dir=""):
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        img_names = sorted(os.listdir(eps_dir))
        start_time = int(img_names[1].split('.')[0].split('_')[0])
        with open(eps_dir+"/0000.txt", "r") as f:
            lines = f.readlines()
        with open(eps_dir+"/0000.txt", "w") as f:
            f.writelines(lines[2*(start_time-1):])



if __name__ == "__main__":
    # [36511   791  1010    27   252]
    # [6338  101   89    0    2]
    # all green [114395   1659   1187    637    460]
    # train_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/train"
    # print stat_obj_action_num(train_dir)
    # test_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/val"
    # print stat_obj_action_num(test_dir)

    # train_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_obj80"
    # print stat_obj_action_num(train_dir)
    #
    # train_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/valid"
    # val_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/valid"
    # rm_obj_start_time_imgs(val_dir)

    # train_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_obj80_docker005_no_early_stopping_all_green/train"
    # train_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/train"
    # num_stat, action3_stat, action4_stat = stat_obj_action_num(train_dir)
    # print num_stat
    # print action3_stat
    # print len(action3_stat)
    # print action4_stat
    # print len(action4_stat)

    # stat_action_obj_start_time(train_dir)

    # rm_obj_start_time_imgs(train_dir)

    # obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/valid"
    # rm_obj_pred_acts(obj_dir)
    # complete_last_three_imgs(obj_dir)

    # test_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/test_rm_txt_start_time"
    # obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/valid"
    # rm_obj_txt_start_time_lines(obj_dir)

    obj_dir = "/home/pirate03/hobotrl_data/A3CCarRecordingDiscrete2/train"
    obj_stats = stat_obj_action_num(obj_dir)
    # [  6910.  21584.  26952.   6100.   9927.  70329.   4678.  12266.  21468.]
    # 180214
    # [ 0.0383433 ,  0.11976872,  0.14955553,  0.03384865,  0.05508451,
    #   0.3902527 ,  0.02595803,  0.06806352,  0.11912504]

    obj_dir = "/home/pirate03/hobotrl_data/A3CCarRecordingDiscrete2/valid"
    obj_stats = stat_obj_action_num(obj_dir)
    # [  365.   900.  1189.   345.   421.  3259.   253.   515.  1001.]
    # 8248
    # [ 0.04425315  0.10911736  0.14415616  0.04182832  0.05104268  0.39512609
    # 0.0306741   0.06243938  0.12136275]
    print obj_stats
    print obj_stats.sum()
    print obj_stats / obj_stats.sum()
