import random
import time

import numpy as np
import tensorflow as tf
from playground.initialD.imitaion_learning.sl.evaluate import evaluate
from playground.initialD.imitaion_learning.process_data.stack_imgs import stack_obj_eps, read_eps_imgs_acts, \
    stack_one_eps
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer


def f_net(inputs, l2):
    """
    action_num is set 5.
    :param inputs:
    :return:
    """
    inputs = inputs[0]
    inputs = inputs/128 - 1.0
    action_num = 5
    # (350, 350, 3*n) -> ()
    conv1 = layers.conv2d(
        inputs=inputs, filters=16, kernel_size=(8, 8), strides=1,
        kernel_regularizer=l2_regularizer(scale=l2),
        activation=tf.nn.relu, name='conv1')
    print conv1.shape
    pool1 = layers.max_pooling2d(
        inputs=conv1, pool_size=3, strides=4, name='pool1')
    print pool1.shape
    conv2 = layers.conv2d(
        inputs=pool1, filters=16, kernel_size=(5, 5), strides=1,
        kernel_regularizer=l2_regularizer(scale=l2),
        activation=tf.nn.relu, name='conv2')
    print conv2.shape
    pool2 = layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=3, name='pool2')
    print pool2.shape
    conv3 = layers.conv2d(
         inputs=pool2, filters=64, kernel_size=(3, 3), strides=1,
         kernel_regularizer=l2_regularizer(scale=l2),
         activation=tf.nn.relu, name='conv3')
    print conv3.shape
    pool3 = layers.max_pooling2d(
        inputs=conv3, pool_size=3, strides=2, name='pool3',)
    print pool3.shape
    depth = pool3.get_shape()[1:].num_elements()
    inputs = tf.reshape(pool3, shape=[-1, depth])
    print inputs.shape
    hid1 = layers.dense(
        inputs=inputs, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=l2), name='hid1')
    print hid1.shape
    hid2 = layers.dense(
        inputs=hid1, units=256, activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=l2), name='hid2')
    print hid2.shape
    pi = layers.dense(
        inputs=hid2, units=action_num, activation=tf.nn.softmax,
        kernel_regularizer=l2_regularizer(scale=l2), name='pi')
    return {"pi": pi}


def split_stack_infos(stack_infos):
    # big bug !!!!!!!!!!!!!!!!!
    # split_infos = [[]] * 5
    # big bug !!!!!!!!!!!!!!!!!
    split_infos = [[] for _ in range(5)]
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


tf.app.flags.DEFINE_string('val_dir', "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/val", """Path to test dataset""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('train_interval', 20, """Train display interval.""")
tf.app.flags.DEFINE_integer('val_interval', 20, """Val display interval.""")
tf.app.flags.DEFINE_integer('val_itr', 1, """Val test number.""")
tf.app.flags.DEFINE_float('l2', 1e-4, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('initial_lr', 1e-2, """Learning rate""")
tf.app.flags.DEFINE_integer('max_step', 600, """Train iteration nums""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.5, """GPU using fraction""")
tf.app.flags.DEFINE_string('log_dir', './log_sl_rnd_stack3_test', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('stack_num', 3, """stack num.""")


FLAGS = tf.app.flags.FLAGS
val_dir = FLAGS.val_dir
batch_size = FLAGS.batch_size
train_interval = FLAGS.train_interval
val_interval = FLAGS.val_interval
val_itr = FLAGS.val_itr
l2 = FLAGS.l2
initial_lr = FLAGS.initial_lr
max_step = FLAGS.max_step
gpu_fraction = FLAGS.gpu_fraction
val_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/val"
log_dir = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/log_sl_rnd_imbalance_1"
stack_num = FLAGS.stack_num


state_shape = (350, 350, 3*stack_num)
labels = [0, 1, 2, 3, 4]
x = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="img")
pi = f_net([x], l2)['pi']
y_ = tf.placeholder(dtype=tf.int32, shape=[None], name="act")
y_one_hot = tf.one_hot(y_, depth=5, name="act_one_hot")
cross_entropy = -tf.reduce_mean(tf.to_float(y_one_hot) * tf.log(pi))
probs = pi
preds = tf.to_int32(tf.argmax(probs, 1))
acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_), "float"))
train_op = tf.train.AdadeltaOptimizer(initial_lr).minimize(cross_entropy)

graph = tf.get_default_graph()
global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

init_op = tf.global_variables_initializer()

sv = tf.train.Supervisor(graph=graph,
                        global_step=global_step,
                        init_op=init_op,
                        summary_op=None,
                        summary_writer=None,
                        logdir=log_dir,)
                        # save_summaries_secs=0)

config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction),
            allow_soft_placement=False,
            # allow_soft_placement=True,
            log_device_placement=False)

val_data = stack_obj_eps(val_dir, stack_num)
data_num = len(val_data)
import os

with sv.managed_session(config=config) as sess:
    tf.train.start_queue_runners(sess)
    # ave_val_loss, ave_val_acc, ave_val_prec, ave_val_rec, ave_val_f1, ave_conf_mat \
    #     = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    obj_labels = []
    obj_preds = []
    obj_dir = "/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_rnd_obj_v3_fenkai_rm_stp/val"
    eps_names = sorted(os.listdir(obj_dir))
    print eps_names
    for eps_name in eps_names:
        time.sleep(0.1)
        eps_dir = obj_dir + "/" + eps_name
        img_names = sorted(os.listdir(eps_dir))[1:]
        eps_stat_txt = open(eps_dir+"/0000.txt", 'r').readlines()
        new_eps_stat_txt = []
        eps_imgs, eps_acts = read_eps_imgs_acts(eps_dir)
        eps_stack_info = stack_one_eps(eps_imgs, eps_acts, stack_num)
        eps_labels = []
        eps_preds = []
        val_imgs, val_labels = concat_imgs_acts(eps_stack_info)
        for i in range(len(val_imgs)):
            time.sleep(0.05)
            val_img = val_imgs[i]
            val_label = val_labels[i]
            val_prob, val_pred, val_loss, val_acc = sess.run([probs, preds, cross_entropy, acc],
                                                             feed_dict={x:np.array([val_img]), y_:np.array([val_label])})
            eps_labels.append(val_label)
            eps_preds.append(val_pred[0])

        for i, val_pred in enumerate(eps_preds):
            os.rename(eps_dir+"/"+img_names[i], eps_dir+"/"+img_names[i]+"_"+str(val_pred))
            new_eps_stat_txt.append(eps_stat_txt[i].split('\n')[0]+","+str(val_pred)+"\n")

        f = open(eps_dir+"/0001.txt", "w")
        for line in new_eps_stat_txt:
            f.write(line)
        f.close()
        prec, rec, f1, conf_mat = evaluate(np.array(eps_labels), np.array(eps_preds), labels)
        obj_preds.extend(eps_preds)
        obj_labels.extend(eps_labels)
        print "eps_name: ", eps_name, "prec: ", prec, "rec: ", rec, "f1: ", f1
        print "conf_mat: \n", conf_mat

    stat_prec, stat_rec, stat_f1, stat_conf_mat = evaluate(np.array(obj_labels), np.array(obj_preds), labels)
    print "stat result"
    print "prec: ", prec, "rec: ", rec, "f1: ", f1
    print "conf_mat: \n", conf_mat





