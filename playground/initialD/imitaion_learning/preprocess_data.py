import os
import tensorflow as tf
from pprint import pprint



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


if __name__ == '__main__':
    stack_info = test_get_all_eps_info()
    pprint(stack_info)
    # test_zero_prefix_files()
    # zero_prefix_recording()