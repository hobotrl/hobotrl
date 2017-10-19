import os
import numpy as np

def stat_eps_action_num(eps_dir):
    filenames = sorted(os.listdir(eps_dir))
    img_names =filenames[1:]
    stat_acts = [0, 0, 0, 0, 0]
    for img_name in img_names:
        act = int(img_name.split('.')[0].split('_')[-1])
        stat_acts[act] += 1
    return np.array(stat_acts)


def stat_obj_action_num(obj_dir):
    eps_names = sorted(os.listdir(obj_dir))
    obj_stats = np.array([0, 0, 0, 0, 0])
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        eps_stats = stat_eps_action_num(eps_dir)
        obj_stats += eps_stats
    return obj_stats


if __name__ == "__main__":
    # [36511   791  1010    27   252]
    # [6338  101   89    0    2]
    train_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/train"
    print stat_obj_action_num(train_dir)
    test_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/val"
    print stat_obj_action_num(test_dir)