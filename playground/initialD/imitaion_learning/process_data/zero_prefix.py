import os


def zero_prefix_files(obj_dir_list):
    for obj_dir in obj_dir_list:
        eps_dirs = os.listdir(obj_dir)
        for eps_dir in eps_dirs:
            os.rename(obj_dir + "/" + eps_dir + "/0.txt", obj_dir + "/" + eps_dir + "/0000.txt")


if __name__ == "__main__":
    obj_dir_list = ["/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards"]
    zero_prefix_files(obj_dir_list)
