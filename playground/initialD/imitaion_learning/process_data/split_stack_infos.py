import random
import numpy as np

def split_stack_infos(stack_infos, num_class=3):
    # big bug !!!!!!!!!!!!!!!!!
    # split_infos = [[]] * 5
    # big bug !!!!!!!!!!!!!!!!!
    split_infos = [[] for _ in range(num_class)]
    for info in stack_infos:
        act = info[-1]
        split_infos[act].append(info)
    return split_infos


def wrap_data(stack_info):
    """
    Due to rnd_imgs_acts is not so convenient when train, so need to wrap data first so that it can easily get stack_imgs
    and acts.
    :param stack_info:
    :return:
    """
    pass


def rand_stack_infos(stack_infos, batch_size):
    """
    :param stack_infos:
           [[img1, img1, img1, action1],
            [img1, img1, img2, action2],
            [img1, img2, img3, action3],
            [img2, img3, img4, action4],
            ......]]
    :param batch_size:
    :return:
    """
    batch_stack_infos = []
    for _ in range(batch_size):
        info = random.choice(stack_infos)
        batch_stack_infos.append(info)
    return batch_stack_infos


def rand_stack_infos_specify_batch_size(splited_stack_infos, batch_size_list):
    assert len(splited_stack_infos) == len(batch_size_list)
    batch_infos = []
    for i in range(len(batch_size_list)):
        if splited_stack_infos[i] == []:
            assert batch_size_list[i] == 0
            # do not sample
            pass
        else:
            batch_info = rand_stack_infos(splited_stack_infos[i], batch_size_list[i])
            batch_infos.extend(batch_info)
    return batch_infos


def concat_imgs_acts(stack_infos):
    stack_imgs = []
    acts = []
    for info in stack_infos:
        imgs = info[:-1]
        act = info[-1]
        stack_imgs.append(np.concatenate(imgs, -1))
        # stack_imgs shape: (none, n, n, 3*stack_num)
        acts.append(act)
    return np.array(stack_imgs), np.array(acts)


def rand_imgs_acts(stack_infos, batch_size):
    batch_infos = rand_stack_infos(stack_infos, batch_size)
    stack_imgs, acts = concat_imgs_acts(batch_infos)
    return stack_imgs, acts


def rand_imgs_acts_specify_batch_size(splited_stack_infos, batch_size_list):
    batch_infos = rand_stack_infos_specify_batch_size(splited_stack_infos, batch_size_list)
    stack_imgs, acts = concat_imgs_acts(batch_infos)
    return stack_imgs, acts