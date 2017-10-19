import os
import numpy as np
import cv2

def cal_hor_sim(li, ri):
    """
    :param li: (n, n)
    :param ri: (n, n)
    :return:
    """
    li0 = np.sum(li, axis=0)+0.0
    li1 = np.sum(li, axis=1)+0.0
    ri0 = np.sum(ri, axis=0)+0.0
    ri1 = np.sum(ri, axis=1)+0.0
    # return (hist_similar(li0, ri0) + hist_similar(li1, ri1)) / 2.0
    return 1.0 - (np.sum(np.abs(li0-ri0))+np.sum(np.abs(li1-ri1))+0.0)/(2.0*np.sum(li0))


def get_eps_divid_point(eps_dir=""):
    img_names = sorted(os.listdir(eps_dir))[1:]
    imgs = []
    sims = []
    for name in img_names:
        imgs.append(cv2.imread(eps_dir + "/" + name, 0))
    for i in range(len(img_names) - 1):
        sims.append(cal_hor_sim(imgs[i], imgs[i + 1]))

    ind = np.where(np.array(sims) <= 0.0)[0]
    ind = list(ind)
    print "ind: ", ind
    print ""
    return ind

def get_obj_divide_point(obj_dir=""):
    eps_names = os.listdir(obj_dir)
    inds = []
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        ind = get_eps_divid_point(eps_dir)
        inds.append(ind)
    print "inds: ", inds
    return inds

def read_eps_imgs(eps_dir):
    img_names = sorted(os.listdir(eps_dir))[1:]
    imgs = []
    for name in img_names:
        imgs.append(cv2.imread(eps_dir + "/" + name, 0))
    return imgs

def read_raw_imgs(eps_dir):
    img_names = sorted(os.listdir(eps_dir))[1:]
    raw_imgs = []
    for name in img_names:
        raw_imgs.append(cv2.imread(eps_dir + "/" + name))
    return raw_imgs

def divide_eps_imgs(imgs):
    sims = []
    for i in range(len(imgs) - 1):
        sims.append(cal_hor_sim(imgs[i], imgs[i + 1]))
    ind = np.where(np.array(sims) <= 0.5)[0]
    ind = list(ind)
    return ind


def mk_new_eps(eps_dir, new_obj_dir, current_eps_num):
    """
    Divide epses from eps_dir to new_obj_dir. current_eps_num means the num of new_obj_dir.
    :param eps_dir:
    :param new_obj_dir:
    :param current_eps_num:
    :return:
    """
    txt = open(eps_dir + "/" + "0000.txt", 'r')
    lines = txt.readlines()

    imgs = read_eps_imgs(eps_dir)
    raw_imgs = read_raw_imgs(eps_dir)
    assert len(lines) == len(imgs)
    ind = divide_eps_imgs(imgs)
    ind.insert(0, -1)
    ind.append(len(imgs)-1)
    new_epss = [raw_imgs[ind[i]+1:ind[i+1]+1] for i in range(len(ind)-1)]
    new_lines = [lines[ind[i]+1:ind[i+1]+1] for i in range(len(ind)-1)]
    for i, new_eps in enumerate(new_epss):
        new_eps_dir = new_obj_dir + "/" + str(current_eps_num+i+1).zfill(4)
        os.mkdir(new_eps_dir)
        txt = open(new_eps_dir + "/0000.txt", 'w')
        eps_lines = new_lines[i]
        txt.writelines(new_lines[i])
        txt.close()
        for j, raw_img in enumerate(new_eps):
            act = eps_lines[j].split(',')[1]
            new_img_path = new_eps_dir + "/" + str(j+1).zfill(4) + "_" + str(act) + ".jpg"
            cv2.imwrite(new_img_path, raw_img)
    return ind[1:-1]


def divide_obj(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3",
               new_obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_new"):
    eps_names = sorted(os.listdir(obj_dir))
    os.mkdir(new_obj_dir)
    current_num = 0
    for i, eps_name in enumerate(eps_names):
        eps_dir = obj_dir + "/" + eps_name
        ind = mk_new_eps(eps_dir, new_obj_dir, current_num)
        divi_num = len(ind)+1
        current_num += divi_num
        print eps_name, ": ", ind


def get_prep_stop_point(img_names):
    """
    there are some biandao before stops if car is in the middle lane, use action 2 and 0. So needs to be filtered for second time.
    :param eps_dir:
    :return:
    """
    # img_names = sorted(os.listdir(eps_dir))
    acts = [img_name.split('.')[0].split('_')[-1] for img_name in img_names]

    cond = 'start'

    zero_max_count = 10
    # zero_count = None

    for i in range(len(acts) - 1, -1, -1):
        act = acts[i]
        if cond == 'start':
            if act == '3':
                cond = 'a'
            # elif act == '0':
            #     cond = 'a0'
            # elif act == '1':
            #     cond = 'a1'
            # elif act == '2':
            #     cond = 'a2'
            else:
                return 'start', i + 1
        elif cond == 'a':
            if act == '3' or act == '4':
                cond = 'a'
            elif act == '2':
                cond = 'b'
            else:
                return 'acc_a', i + 1
        elif cond == 'a0':
            if act == '0' or act == '4':
                cond = 'a0'
            elif act == '2':
                cond = 'b'
            else:
                return 'acc_a0', i + 1
        elif cond == 'b':
            if act == '2' or act == '4':
                cond = 'b'
            elif act == '0':
                cond = 'c'
                zero_count = 1
            elif act == '1':
                return 'end_b', i + 1
            else:
                return 'acc_b', i + 1
        elif cond == 'c':
            if act == '0' or act == '4':
                cond = 'c'
                zero_count += 1
                if zero_count >= zero_max_count:
                    return 'end2_c', i + zero_max_count
            elif act == '1':
                return 'end_c', i + 1
            elif act == '2':
                cond = 'd'
            else:
                return 'acc_c', i + 1
        elif cond == 'd':
            if act == '2' or act == '4':
                cond = 'd'
            elif act == '0':
                cond = 'e'
                zero_count = 1
            elif act == '1':
                return 'end_d', i + 1
            else:
                return 'acc_d', i + 1
        elif cond == 'e':
            if act == '0' or act == '4':
                cond = 'e'
                zero_count += 1
                if zero_count >= zero_max_count:
                    return 'end2_e', i + zero_max_count
            elif act == '2':
                cond = 'f'
            elif act == '1':
                return 'end_e', i + 1
            else:
                return 'acc_e', i + 1
        elif cond == 'f':
            if act == '2' or act == '4':
                cond = 'f'
            elif act == '0' or act == '1':
                return 'end_f', i + 1
            else:
                return 'acc_f', i + 1
        else:
            return "yiwai1", i + 1

    return "yiwai2_" + cond, i


def mk_jieduan_eps_dir(eps_dir="", new_eps_dir=""):
    os.mkdir(new_eps_dir)

    file_names = sorted(os.listdir(eps_dir))
    txt_name = file_names[0]
    img_names = file_names[1:]
    cond, stop_point = get_prep_stop_point(img_names)

    txt = open(eps_dir + "/" + txt_name, 'r')
    lines = txt.readlines()
    new_txt = open(new_eps_dir+"/"+"0000.txt", 'w')
    new_txt.writelines(lines[:stop_point])
    new_txt.close()

    for i, name in enumerate(img_names):
        if i >= stop_point:
            break
        img = cv2.imread(eps_dir+"/"+name)
        cv2.imwrite(new_eps_dir+"/"+name, img)

    return stop_point


def mk_jieduan_obj_dir(obj_dir="", new_obj_dir=""):
    os.mkdir(new_obj_dir)
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        new_eps_dir = new_obj_dir + "/" + eps_name
        mk_jieduan_eps_dir(eps_dir, new_eps_dir)


def little_frames(obj_dir=""):
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        file_num = len(os.listdir(eps_dir))
        if file_num <= 3:
            print eps_name


def get_stop_prep_point_obj(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_new"):
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        file_names = sorted(os.listdir(eps_dir))
        img_names = file_names[1:]
        cond, stop_point = get_prep_stop_point(img_names)
        print eps_name, ": ", stop_point


if __name__ == '__main__':
    # divide_obj("/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80",
    #            "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80_fenkai")
    # divide_obj()
    # mk_jieduan_obj_dir(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_new",
    #                    new_obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_new_rm_stp")
    # little_frames(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_new_rm_stp")
    # get_stop_prep_point_obj()
    print get_obj_divide_point(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rnd_acts_obj80")