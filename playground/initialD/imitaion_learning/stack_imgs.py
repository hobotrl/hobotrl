import os
import tensorflow as tf
import numpy as np
import cv2


def read_eps_imgs_acts(eps_dir):
    filenames = sorted(os.listdir(eps_dir))
    img_names = filenames[1:]
    imgs = []
    acts = []
    for img_name in img_names:
        img_path = eps_dir + "/" +img_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        act = int(img_name.split('.')[0].split('_')[-1])
        acts.append(act)
    return imgs, acts


def stack_one_eps(eps_imgs, eps_acts, stack_num):
    # is there need to complete starting and ending frames?
    """
    :param eps_imgs: numpy array of shape (l, n, n, 3).
    :param eps_acts: numpy array of shape (l,).
    :param stack_num:
    :return: [[img1, img1, img1, action1],
              [img1, img1, img2, action2],
              [img1, img2, img3, action3],
              [img2, img3, img4, action4],
              ......]
              action: int
    """
    assert len(eps_imgs) == len(eps_acts)
    for i in range(stack_num-1):
        eps_imgs.insert(0, eps_imgs[0])
        eps_acts.insert(0, eps_acts[0])
    img_num = len(eps_imgs)
    stack_info = []
    for i in range(img_num-stack_num+1):
        info = []
        for j in range(stack_num):
            info.append(eps_imgs[i+j])
        action = eps_acts[i+stack_num-1]
        info.append(action)
        stack_info.append(info)
    return stack_info


def test_stack_one_eps(eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/test_new_rm_stp/0001",
                          stack_num=3,
                          new_eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/test_stack"):
    os.mkdir(new_eps_dir)
    eps_imgs, eps_acts = read_eps_imgs_acts(eps_dir)
    stack_infos = stack_one_eps(eps_imgs, eps_acts, stack_num)
    f = open(new_eps_dir+"/"+"0000.txt", "w")
    for i, info in enumerate(stack_infos):
        imgs , act = info[0:3], info[3]
        for j, img in enumerate(imgs):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(new_eps_dir+"/"+str(i+1).zfill(4)+"_"+str(j)+".jpg", img)
            f.write(str(i+1)+": "+str(act)+"\n")
    f.close()


def stack_obj_eps(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/test_new_rm_stp",
                  stack_num=3):
    eps_names = sorted(os.listdir(obj_dir))
    obj_stack_info = []
    for i, eps_name in enumerate(eps_names):
        eps_dir = obj_dir + "/" + eps_name
        eps_imgs, eps_acts = read_eps_imgs_acts(eps_dir)
        if len(eps_imgs) >= stack_num:
            eps_stack_info = stack_one_eps(eps_imgs, eps_acts, stack_num)
            # obj_stack_info.append(eps_stack_info)
            obj_stack_info.extend(eps_stack_info)
    return obj_stack_info


def test_stack_obj_eps(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/test_new_rm_stp",
                       new_obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/test_new_rm_stp/stack",
                       stack_num=3):
    os.mkdir(new_obj_dir)


if __name__ == "__main__":
    test_stack_one_eps()
