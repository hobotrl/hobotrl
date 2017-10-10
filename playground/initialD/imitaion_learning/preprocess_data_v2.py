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


def divide_eps(eps_dir=""):
    img_names = sorted(os.listdir(eps_dir))[1:]
    imgs = []
    sims = []
    for name in img_names:
        imgs.append(cv2.imread(eps_dir + "/" + name, 0))
    for i in range(len(img_names) - 1):
        sims.append(cal_hor_sim(imgs[i], imgs[i + 1]))

    ind = np.where(np.array(sims) <= 0.5)[0]
    ind = list(ind)
    return ind


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
    txt = open(eps_dir + "/" + "000000.txt", 'r')
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
        txt.writelines(new_lines[i])
        txt.close()
        for j, raw_img in enumerate(new_eps):
            new_img_path = new_eps_dir + "/" + str(j+1).zfill(4) + ".jpg"
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






if __name__ == '__main__':
    divide_obj()