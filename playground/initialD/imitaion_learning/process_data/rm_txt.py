import os


obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/val"
eps_names = sorted(os.listdir(obj_dir))
for eps_name in eps_names:
    eps_rm_txt = obj_dir + "/" + eps_name + "/" + "0000.txt"
    os.remove(eps_rm_txt)
    # os.rename(obj_dir + "/" + eps_name + "/" + "0001.txt", obj_dir + "/" + eps_name + "/" + "0000.txt")


def rm_txt_start_time(eps_dir):
    txt_name = eps_dir + "/0000.txt"
    f = open(txt_name, "r")
    lines = f.readlines()
    os.remove()
