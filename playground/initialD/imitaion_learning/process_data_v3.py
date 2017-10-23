import os
import numpy as np
import cv2
import shutil

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
    # return 1.0 - (np.sum(np.abs(li0-ri0))+np.sum(np.abs(li1-ri1))+0.0)/(2.0*np.sum(li0))
    # return 1.0 - (np.sum(np.abs(li0-ri0))+np.sum(np.abs(li1-ri1))+0.0)/(np.sum(li0) + np.sum(ri0))
    return 1.0 - (np.sum(np.abs(li0-ri0))+np.sum(np.abs(li1-ri1))+0.0)/min((np.sum(li0),np.sum(ri0)))


def read_eps_gray_imgs(eps_dir):
    img_names = sorted(os.listdir(eps_dir))[1:]
    imgs = []
    for img_name in img_names:
        imgs.append(cv2.imread(eps_dir + "/" + img_name, 0))
    return imgs


def read_eps_raw_imgs(eps_dir, BGR=True):
    img_names = sorted(os.listdir(eps_dir))[1:]
    imgs = []
    for img_name in img_names:
        imgs.append(cv2.imread(eps_dir + "/" + img_name))

    return imgs


def get_conflict_point(imgs, threshold=0.2):
    sims = []
    for i in range(len(imgs) - 1):
        sims.append(cal_hor_sim(imgs[i], imgs[i + 1]))
    ind = np.where(np.array(sims) <= threshold)[0]
    ind = list(ind)
    return ind


def mk_new_eps(eps_dir, new_eps_dir, resize_shape=(256, 256), threshold=0.2):
    img_names = sorted(os.listdir(eps_dir))[1:]
    gray_imgs = read_eps_gray_imgs(eps_dir)
    raw_imgs = read_eps_raw_imgs(eps_dir)
    ind = get_conflict_point(gray_imgs, threshold)
    if ind == [] and len(img_names) >= 5:
        os.mkdir(new_eps_dir)
        shutil.copy(eps_dir+"/0000.txt", new_eps_dir+"/0000.txt")
        for img_name, img in zip(img_names, raw_imgs):
            new_img = cv2.resize(img, resize_shape)
            cv2.imwrite(new_eps_dir + "/" + img_name, new_img)


def mk_new_obj(obj_dir, new_obj_dir, resize_shape=(256, 256), threshold=-0.2):
    os.mkdir(new_obj_dir)
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        new_eps_dir = new_obj_dir + "/" + eps_name
        mk_new_eps(eps_dir, new_eps_dir, resize_shape, threshold)


if __name__ == "__main__":
    # obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_obj80_docker005_no_early_stopping_all_green"
    obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_obj80_docker005_no_early_stopping_all_green2"
    # new_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_docker005_no_early_stopping_all_green"
    new_obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_docker005_no_early_stopping_all_green2"
    mk_new_obj(obj_dir, new_obj_dir)