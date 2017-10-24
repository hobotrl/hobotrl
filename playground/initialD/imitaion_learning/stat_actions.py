import os
import numpy as np

def stat_eps_action_num(eps_dir):
    filenames = sorted(os.listdir(eps_dir))
    img_names =filenames[1:]
    stat_acts = [0, 0, 0, 0, 0]
    has_action3_imgs = []
    has_action4_imgs = []
    for img_name in img_names:
        act = int(img_name.split('.')[0].split('_')[-1])
        if act == 3:
            has_action3_imgs.append(img_name)
        if act == 4:
            has_action4_imgs.append(img_name)
        stat_acts[act] += 1
    return np.array(stat_acts), has_action3_imgs, has_action4_imgs


def stat_obj_action_num(obj_dir):
    eps_names = sorted(os.listdir(obj_dir))
    obj_stats = np.array([0, 0, 0, 0, 0])
    stat_action3_eps = {}
    stat_action4_eps = {}
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        eps_stats, has_action3_imgs, has_action4_imgs = stat_eps_action_num(eps_dir)
        obj_stats += eps_stats
        if has_action3_imgs != []:
            stat_action3_eps[eps_name] = has_action3_imgs
        if has_action4_imgs != []:
            stat_action4_eps[eps_name] = has_action4_imgs
    return obj_stats, stat_action3_eps, stat_action4_eps


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

    obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/valid"
    rm_obj_pred_acts(obj_dir)
    # complete_last_three_imgs(obj_dir)
