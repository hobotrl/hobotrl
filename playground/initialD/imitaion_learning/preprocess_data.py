import os
import tensorflow as tf
from pprint import pprint
import numpy as np
import cv2

def get_image_list():
    pass

def get_image_tensor(image_list):
    filename_queue = tf.train.string_input_producer(image_list)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    example = tf.image.decode_jpeg(value)
    return example

def sess_run_read_images():
    image_list = get_image_list()
    image = get_image_tensor(image_list)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image = sess.run([image])
        print image

        coord.request_stop()
        coord.join(threads)


def get_one_eps_info(cur_dir, stack_num):
    # is there need to complete starting and ending frames?
    """
    :param cur_dir: names of files under cur_dir should be zeros-prefix so that files could be sorted correctly
                    "00001.jpg", "00002.jpg", etc
    :param stack_num:
    :return: [[img1_path, img2_path, img3_path, action3],
              [img2_path, img3_path, img4_path, action4],
              ......]
              img_path: string, "path_to_cur_dir/0001.jpg", "path_to_cur_dir/0012.jpg", etc
              action: int
    """
    filenames = sorted(os.listdir(cur_dir))
    for i in range(stack_num):
        filenames.insert(0, filenames[0])
    file_num = len(filenames)
    stack_info = []
    for i in range(file_num-stack_num):
        info = []
        for j in range(stack_num):
            filepath = cur_dir + '/' + filenames[i+j]
            info.append(filepath)
        action = int(filenames[i+stack_num-1].split('.')[0].split('_')[-1])
        info.append(action)
        stack_info.append(info)
    return stack_info


def test_get_one_eps_info():
    cur_dir = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/exp/record_rule_scenes_rnd_obj_40/1"
    stack_num = 3
    stack_info = get_one_eps_info(cur_dir, stack_num)
    pprint(stack_info)


def get_all_eps_info(obj_dir, stack_num):
    """
    :param obj_dir: names of dirs under obj_dir should be zeros-prefix so that dirs could be sorted correctly
                    "00001", "00002", etc
    :param stack_num:
    :return: [[img1_path, img2_path, img3_path, action3],
              [img2_path, img3_path, img4_path, action4],
              ......]
              img_path: string, "path_to_cur_dir/0001.jpg", "path_to_cur_dir/0012.jpg", etc
              action: int
    """
    eps_dir_names = sorted(os.listdir(obj_dir))
    stack_info = []
    for i in range(len(eps_dir_names)):
        eps_dir_path = obj_dir + "/" + eps_dir_names[i]
        stack_info.extend(get_one_eps_info(eps_dir_path, stack_num))
    return stack_info


def test_get_all_eps_info():
    obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_100"
    stack_num = 3
    return get_all_eps_info(obj_dir, stack_num)


def zero_prefix_files(obj_dir_list):
    for obj_dir in obj_dir_list:
        eps_dirs = os.listdir(obj_dir)
        for eps_dir in eps_dirs:
            for filename in os.listdir(obj_dir + "/" + eps_dir):
                os.rename(obj_dir + "/" + eps_dir + "/" + filename, obj_dir + "/" + eps_dir + "/" + filename.zfill(10))
            os.rename(obj_dir + "/" + eps_dir, obj_dir + "/" + eps_dir.zfill(4))


def test_zero_prefix_files():
    obj_dir_list = ["/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_docker03_rnd_obj_100"]
    zero_prefix_files(obj_dir_list)

def zero_prefix_recording():
    obj_dir_list = ["/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_40",
                    "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_60",
                    "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_80",
                    "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_100"]
    zero_prefix_files(obj_dir_list)


def get_stop_point(eps_dir):
    """
    get the stop time point of car
    :param eps_dir:
    :return:
    """
    img_names = sorted(os.listdir(eps_dir))
    acts = [img_name.split('.')[0].split('_')[-1] for img_name in img_names]
    for i in range(len(acts)-1, -1, -1):
        if acts[i] != '2' and acts[i] != '3':
            break
    return i+1


def test_get_stop_point(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_40"):
    for eps_dir in sorted(os.listdir(obj_dir)):
        print eps_dir, ": ", get_stop_point(obj_dir+"/"+eps_dir)


def filter_stop_frames(obj_dir, new_obj_dir):
    os.mkdir(new_obj_dir)
    for eps_dir in os.listdir(obj_dir):
        os.mkdir(new_obj_dir+"/"+eps_dir)
        stop_point = get_stop_point(obj_dir+"/"+eps_dir)
        imgs = sorted(os.listdir(obj_dir+"/"+eps_dir))
        for i in range(stop_point):
            print "write img: ", new_obj_dir+"/"+eps_dir+"/"+imgs[i]
            targ = open(new_obj_dir+"/"+eps_dir+"/"+imgs[i], "wb")
            src = open(obj_dir+"/"+eps_dir+"/"+imgs[i], "rb")
            targ.write(src.read())
            targ.close()
            src.close()


def test_fileter_stop_frames(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_40",
                             new_obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/filt_record_rule_scenes_rnd_obj_40"):
    filter_stop_frames(obj_dir, new_obj_dir)


def filter_stop_frames_for_all():
    obj_dir_list = ["/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_40",
                    "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_60",
                    "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_80",
                    "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_100"]
    new_obj_dir_list = ["/home/pirate03/hobotrl_data/playground/initialD/exp/filt_record_rule_scenes_rnd_obj_40",
                        "/home/pirate03/hobotrl_data/playground/initialD/exp/filt_record_rule_scenes_rnd_obj_60",
                        "/home/pirate03/hobotrl_data/playground/initialD/exp/filt_record_rule_scenes_rnd_obj_80",
                        "/home/pirate03/hobotrl_data/playground/initialD/exp/filt_record_rule_scenes_rnd_obj_100"]
    for i in range(4):
        filter_stop_frames(obj_dir_list[i], new_obj_dir_list[i])


def get_prep_stop_point(eps_dir):
    """
    there are some biandao before stops if car is in the middle lane, use action 2 and 0. So needs to be filtered for second time.
    :param eps_dir:
    :return:
    """
    img_names = sorted(os.listdir(eps_dir))
    acts = [img_name.split('.')[0].split('_')[-1] for img_name in img_names]

    cond = 'start'

    zero_max_count = 10
    # zero_count = None

    for i in range(len(acts)-1, -1, -1):
        act = acts[i]
        if cond == 'start':
            if act == '3':
                cond = 'a'
            else:
                return 'acc_start', i+1
        elif cond == 'a':
            if act == '3' or act == '4':
                cond = 'a'
            elif act == '2':
                cond = 'b'
            else:
                return 'acc_a', i+1
        elif cond == 'b':
            if act == '2' or act == '4':
                cond = 'b'
            elif act == '0':
                cond = 'c'
                zero_count = 1
            elif act == '1':
                return 'end_b', i+1
            else:
                return 'acc_b', i+1
        elif cond == 'c':
            if act == '0' or act == '4':
                cond = 'c'
                zero_count += 1
                if zero_count >= zero_max_count:
                    return 'end2_c', i+zero_max_count
            elif act == '1':
                return 'end_c', i+1
            elif act == '2':
                cond = 'd'
            else:
                return 'acc_c', i+1
        elif cond == 'd':
            if act == '2' or act == '4':
                cond = 'd'
            elif act == '0':
                cond = 'e'
                zero_count = 1
            elif act == '1':
                return 'end_d', i+1
            else:
                return 'acc_d', i+1
        elif cond == 'e':
            if act == '0' or act == '4':
                cond = 'e'
                zero_count += 1
                if zero_count >= zero_max_count:
                    return 'end2_e', i+zero_max_count
            elif act == '2':
                cond = 'f'
            elif act == '1':
                return 'end_e', i+1
            else:
                return 'acc_e', i+1
        elif cond == 'f':
            if act == '2' or act == '4':
                cond = 'f'
            elif act == '0' or act == '1':
                return 'end_f', i+1
            else:
                return 'acc_f', i+1
        else:
            return "yiwai1", i+1

    return "yiwai2_"+cond, i




def test_get_prep_stop_point(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_part_v2/obj_40",
                             cond_ref='acc'):
    info = []
    for count, eps_dir in enumerate(sorted(os.listdir(obj_dir))):
        if count >= 200:
            break
        cond, i = get_prep_stop_point(obj_dir+"/"+eps_dir)
        if cond_ref in cond:
            print eps_dir, ": ", cond, ", ", i
            info.append([cond, i])
    print len(info)
    # cond, i = get_prep_stop_point(obj_dir + "/" + '0032')
    # print '0032', ": ", cond, " ", i


def diff(eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_part/obj_100/0016"):
    img_names = sorted(os.listdir(eps_dir))
    imgs = []
    for name in img_names:
        imgs.append(cv2.imread(eps_dir+"/"+name))
    for i in range(len(img_names)-1):
        print i, ": ", np.linalg.norm(imgs[i]-imgs[i+1])



if __name__ == '__main__':
    # filter_stop_frames_for_all()
    # stack_info = test_get_all_eps_info()
    # pprint(stack_info)
    # test_zero_prefix_files()
    # zero_prefix_recording()
    # test_get_stop_point()
    test_get_prep_stop_point()
    # diff()