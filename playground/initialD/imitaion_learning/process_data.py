import os
import tensorflow as tf
from pprint import pprint
import numpy as np
import cv2
import Image


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
    # obj_dir_list = ["/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_docker03_rnd_obj_100"]
    obj_dir_list = ["/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_v3"]

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
            elif act == '0':
                cond = 'a0'
            # elif act == '1':
            #     cond = 'a1'
            # elif act == '2':
            #     cond = 'a2'
            else:
                return 'acc_start', i+1
        elif cond == 'a':
            if act == '3' or act == '4':
                cond = 'a'
            elif act == '2':
                cond = 'b'
            else:
                return 'acc_a', i+1
        elif cond == 'a0':
            if act == '0' or act == '4':
                cond = 'a0'
            elif act == '2':
                cond = 'b'
            else:
                return 'acc_a0', i+1
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


def rename_dirs(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_v3",
                start_num=0):
    eps_names = sorted(os.listdir(obj_dir))
    for i, name in enumerate(eps_names):
        eps_dir = os.path.join(obj_dir, name)
        new_eps_dir = os.path.join(obj_dir, str(start_num+i+1).zfill(4))
        os.rename(eps_dir, new_eps_dir)


def diff(eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_part/obj_100/0016"):
    img_names = sorted(os.listdir(eps_dir))
    imgs = []
    diffs = []
    for name in img_names:
        imgs.append(cv2.imread(eps_dir+"/"+name))
    for i in range(len(img_names)-1):
        diffs.append(np.linalg.norm(imgs[i]/255.0-imgs[i+1]/255.0))
    return diffs


def test_diff(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_part_v2/obj_40"):
    eps_names = sorted(os.listdir(obj_dir))
    for eps_name in eps_names:
        eps_dir = obj_dir + "/" + eps_name
        diffs = diff(eps_dir)
        mean_diff = np.mean(np.array(diffs))
        print "eps: ", eps_name
        for i, di in enumerate(diffs):
            print "i: ", i, "diff: ", di
        print "mean_diff: ", mean_diff
        print "===========\n"*3


def split_image(img, part_size=(175, 175)):
    # w, h = img.shape
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i+pw, j+ph)).copy()
            for i in xrange(0, w, pw)
            for j in xrange(0, h, ph)]


def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else abs(float(l)-float(r)))/max(float(l), float(r), 1.0) for l, r in zip(lh, rh))/len(lh)


def calc_similar(li, ri):
    # return sum(hist_similar(cv2.calcHist([l], [0], None, [256], [0, 256]),
    #                         cv2.calcHist([r], [0], None, [256], [0, 256]))
    #            for l, r in zip(split_image(li), split_image(ri))) / 25.0
    return sum(hist_similar(l.histogram(), r.histogram())
               for l, r in zip(split_image(li), split_image(ri))) / 4.0

def calc_similar_eps(eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_orig/80/0084"):
    img_names = sorted(os.listdir(eps_dir))
    imgs = []
    sims = []
    for name in img_names:
        # imgs.append(cv2.imread(eps_dir + "/" + name, 0))
        imgs.append(Image.open(eps_dir+"/"+name))
    for i in range(len(img_names) - 1):
        sims.append(calc_similar(imgs[i], imgs[i+1]))
    return sims

def calc_similar_obj(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_part_v2/obj_80"):
    names = sorted(os.listdir(obj_dir))
    for name in names:
        simis = calc_similar_eps(obj_dir+"/"+name)
        ind = np.where(np.array(simis)<=0.46)
        print name, ": ", ind, np.array(simis)[ind]


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


def cal_hor_sim_eps(eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_orig/80/0084"):
    img_names = sorted(os.listdir(eps_dir))
    imgs = []
    sims = []
    for name in img_names:
        imgs.append(cv2.imread(eps_dir + "/" + name, 0))
    for i in range(len(img_names) - 1):
        sims.append(cal_hor_sim(imgs[i], imgs[i+1]))
    # for i in range(len(sims)):
    #     if sims[i] < 0.5:
    #         print i+1, sims[i]
    return sims


def cal_hor_sim_obj(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_part_v2/obj_80"):
    names = sorted(os.listdir(obj_dir))
    for name in names:
        simis = cal_hor_sim_eps(obj_dir + "/" + name)
        ind = np.where(np.array(simis) <= 0.5)
        print name, ": ", ind, np.array(simis)[ind]


def divide_eps(eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_orig/80/0084"):
    # well need to filter .txt
    img_names = sorted(os.listdir(eps_dir))
    imgs = []
    sims = []
    for name in img_names:
        imgs.append(cv2.imread(eps_dir + "/" + name, 0))
    for i in range(len(img_names) - 1):
        sims.append(cal_hor_sim(imgs[i], imgs[i+1]))
    ind = np.where(np.array(sims) <= 0.5)[0]
    ind = list(ind)
    ind.append(len(imgs)-1)
    # division_imgs = [imgs[ind[i]:ind[i+1]] for i in ind]
    division_names = [img_names[ind[i]+1:ind[i+1]+1] for i in range(len(ind)-1)]
    # division_names.append(img_names[-1])
    return division_names

def test_divie_eps(eps_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_orig/80/0084"):
    names = divide_eps(eps_dir)
    for name in names:
        print name

def divide_obj(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_v3"):
    eps_names = sorted(os.listdir(obj_dir))
    current_eps_num = len(eps_names)
    for name in eps_names:
        eps_dir = obj_dir + "/" + name
        img_names_list = divide_eps(eps_dir)
        for img_names in img_names_list:
            current_eps_num += 1
            new_eps_name = str(current_eps_num).zfill(4)
            new_eps_dir = os.path.join(obj_dir, new_eps_name)
            os.mkdir(new_eps_dir)
            for i, img_name in enumerate(img_names):
                img_path = os.path.join(eps_dir, img_name)
                new_img_path = os.path.join(new_eps_dir, str(i).zfill(4)+".jpg")
                os.rename(img_path, new_img_path)


def test_divide_obj(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/test_"):
    divide_obj(obj_dir)


if __name__ == '__main__':
    # filter_stop_frames_for_all()
    # stack_info = test_get_all_eps_info()
    # pprint(stack_info)
    # test_zero_prefix_files()
    # zero_prefix_recording()
    # test_get_stop_point()
    # test_get_prep_stop_point()
    # diff()
    # sims = calc_similar_eps()
    # for i in range(len(sims)):
    #     if sims[i] < 0.5:
    #         print "i: ", i+1, sims[i]
    # calc_similar_obj()
    # cal_hor_sim_eps()
    # cal_hor_sim_obj(obj_dir="/home/pirate03/hobotrl_data/playground/initialD/exp/test_")
    # test_divie_eps("/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_v3/0001")
    obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_vec_rewards_docker005_no_early_stopping_all_green2"
    rename_dirs(obj_dir, start_num=550)
    # divide_obj()
    # test_divide_obj()