import tensorflow as tf
from stack_imgs import stack_obj_eps
import numpy as np
import random
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer
from evaluate import evaluate
import time
from datetime import datetime


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


def rnd_imgs_acts(stack_infos, batch_size):
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
    stack_imgs = []
    acts = []
    for _ in range(batch_size):
        info = random.choice(stack_infos)
        imgs = info[:-1]
        act = info[-1]
        stack_imgs.append(np.concatenate(imgs, -1))
        acts.append(act)
    return np.array(stack_imgs), np.array(acts)


# def splited_rnd_imgs_acts(splited_stack_infos):
#     num = len(splited_stack_infos)
#     assert num == 5
#     num0 = 100
#     num1 = 30
#     num2 = 30
#     num3 = 10
#     num4 = 10
#     imgs0, acts0 = rnd_imgs_acts(splited_stack_infos[0], num0)
#     imgs1, acts1 = rnd_imgs_acts(splited_stack_infos[1], num1)
#     imgs2, acts2 = rnd_imgs_acts(splited_stack_infos[2], num2)
#     imgs3, acts3 = rnd_imgs_acts(splited_stack_infos[3], num3)
#     imgs4, acts4 = rnd_imgs_acts(splited_stack_infos[4], num4)
#     imgs = np.concatenate((imgs0, imgs1, imgs2, imgs3, imgs4), axis=0)
#     acts = np.concatenate((acts0, acts1, acts2, acts3, acts4), axis=0)
#     return imgs, acts


tf.app.flags.DEFINE_string('train_dir', "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/train1", """Path to training dataset""")
tf.app.flags.DEFINE_string('val_dir', "/home/pirate03/hobotrl_data/playground/initialD/exp/TEST/val1", """Path to test dataset""")
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('train_interval', 20, """Train display interval.""")
tf.app.flags.DEFINE_integer('val_interval', 20, """Val display interval.""")
tf.app.flags.DEFINE_integer('val_itr', 5, """Val test number.""")
tf.app.flags.DEFINE_float('l2', 1e-4, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('initial_lr', 1e-3, """Learning rate""")
tf.app.flags.DEFINE_float('max_step', 200, """Train iteration nums""")
tf.app.flags.DEFINE_string('log_dir', './log_sl_rnd_stack1', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_string('stack_num', 1, """Directory where to write log and checkpoint.""")


FLAGS = tf.app.flags.FLAGS
train_dir = FLAGS.train_dir
val_dir = FLAGS.val_dir
batch_size = FLAGS.batch_size
train_interval = FLAGS.train_interval
val_interval = FLAGS.val_interval
val_itr = FLAGS.val_itr
l2 = FLAGS.l2
initial_lr = FLAGS.initial_lr
max_step = int(FLAGS.max_step)
log_dir = FLAGS.log_dir
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
                        logdir=log_dir,
                        save_summaries_secs=0)

config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
            # allow_soft_placement=False,
            allow_soft_placement=True,
            log_device_placement=False)

train_data = stack_obj_eps(train_dir, stack_num)
val_data = stack_obj_eps(val_dir, stack_num)
# train_data_splited = split_stack_infos(train_data)
# val_data_splited = split_stack_infos(val_data)

with sv.managed_session(config=config) as sess:
    # init_step = global_step.eval(sess=sess)
    tf.train.start_queue_runners(sess)
    for step in xrange(0, max_step):
        if step % val_interval == 0:
            print "==========val %d=========" %step
            ave_val_loss, ave_val_acc, ave_val_prec, ave_val_rec, ave_val_f1, ave_conf_mat\
                = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for i in range(val_itr):
                val_imgs, val_labels = rnd_imgs_acts(val_data, batch_size)
                # val_imgs, val_labels = splited_rnd_imgs_acts(val_data_splited)
                val_preds, val_loss, val_acc = sess.run([preds, cross_entropy, acc], feed_dict={x: val_imgs, y_: val_labels})
                prec, rec, f1, conf_mat = evaluate(val_labels, val_preds, labels)
                ave_val_loss += val_loss
                ave_val_acc += val_acc
                ave_val_prec += prec
                ave_val_rec += rec
                ave_val_f1 += f1
                ave_conf_mat += conf_mat
            ave_val_loss /= val_itr
            ave_val_acc /= val_itr
            ave_val_prec /= val_itr
            ave_val_rec /= val_itr
            ave_val_f1 /= val_itr
            ave_conf_mat /= val_itr
            print "loss: ", ave_val_loss, "acc: ", ave_val_acc
            print "prec: ", ave_val_prec
            print "rec:  ", ave_val_rec
            print "conf_mat: "
            print ave_conf_mat
        # lr_value = get_lr(initial_lr, lr_decay, lr_decay_steps, global_step)
        start_time = time.time()
        train_imgs, train_labels = rnd_imgs_acts(train_data, batch_size)
        # train_imgs, train_labels = splited_rnd_imgs_acts(train_data_splited)
        _, train_preds, train_loss, train_acc = sess.run([train_op, preds, cross_entropy, acc],
                                                         feed_dict={x:train_imgs, y_:train_labels})
        duration = time.time() - start_time
        if step % train_interval == 0 or step < 10:
            # print "==========train %d========" %step
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: (Training) step %d, loss=%.4f, '
                          'acc=%.4f, '
                          # 'lr=%f '
                          '(%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, train_loss,
                                 train_acc,
                                 # lr_value,
                                 examples_per_sec, sec_per_batch))
            prec, rec, f1, conf_mat = evaluate(train_labels, train_preds, labels)
            # print "loss: ", train_loss, "acc: ", train_acc
            print "prec: ", prec
            print "rec: ", rec
            print "conf_mat: "
            print conf_mat

