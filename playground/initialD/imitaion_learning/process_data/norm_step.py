import os

obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/val"
eps_names = sorted(os.listdir(obj_dir))
for eps_name in eps_names:
    eps_dir = obj_dir + "/" + eps_name
    lines = open(eps_dir+"/0000.txt", "r").readlines()
    new_txt = open(eps_dir+"/0001.txt", "w")
    for i, line in enumerate(lines):
        line.split(",")[0] = str(i+1)
        new_txt.write(line)
    new_txt.close()