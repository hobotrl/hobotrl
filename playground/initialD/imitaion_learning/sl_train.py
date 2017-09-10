import sys
sys.path.append('../../..')

import hobotrl as hrl
import tensorflow as tf
import hobotrl.network as network
from initialD_input import distorted_inputs
sys.path.append("/home/pirate03/anaconda2/lib/python2.7/site-packages")

import sklearn.metrics
import os

def f_net(inputs):
    l2 = 1e-3
    state = inputs[0]
    conv = hrl.utils.Network.conv2ds(state, shape=[(32, 4, 4), (64, 4, 4), (64, 2, 2)], out_flatten=True,
                                     activation=tf.nn.relu,
                                     l2=l2, var_scope="convolution")
    q = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
                                    l2=l2, var_scope="q")
    pi = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
                                     activation_out=tf.nn.softmax, l2=l2, var_scope="pi")
    return {"q": q, "pi": pi}

def evaluate(y_true, preds):
    prec = sklearn.metrics.precision_score(y_true, preds, average=None)
    rec = sklearn.metrics.recall_score(y_true, preds, average=None)
    f1 = sklearn.metrics.f1_score(y_true, preds, average=None)
    conf_mat = sklearn.metrics.confusion_matrix(y_true, preds)
    return prec, rec, f1, conf_mat
#
# def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
#     lr = initial_lr
#     for s in lr_decay_steps:
#         if global_step >= s:
#             lr *= lr_decay
#     return lr




# construct network
state_shape = (224, 224, 3)
x = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="image")
# networks = network.Network([x], f_net, var_scope="learn")
# q = networks['q']
# pi = networks['pi']
pi_q = f_net([x])
pi = pi_q['pi']
# construct loss and train operator
y_ = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
y_one_hot = tf.one_hot(y_, depth=3, name="y_one_hot")
cross_entropy = -tf.reduce_mean(tf.to_float(y_one_hot)*tf.log(pi))
probs = pi
preds = tf.to_int32(tf.argmax(probs, 1))
print "preds: ", preds
# acc = tf.reduce_mean(tf.equal(y_, preds))
acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_), "float"))
train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(cross_entropy)

# set hyper parameters
logdir = "./sl_fnet_train4"
train_dataset = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/train.tfrecords"
val_dataset = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/filter_action3/val.tfrecords"
val_interval = 200
train_interval = 100
val_itr = 10
batch_size = 256
lr_decay_steps = 20
max_step = 10000
# initial_lr = 0.01
# lr_decay = 0.01

# construct data input
train_images, train_labels = distorted_inputs(train_dataset, batch_size, num_threads=4)
val_images, val_labels = distorted_inputs(val_dataset, batch_size, num_threads=4)

# set session parameters
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
                        logdir=logdir,
                        save_summaries_secs=0)

config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7),
            # allow_soft_placement=False,
            allow_soft_placement=True,
            log_device_placement=False)

with sv.managed_session(config=config) as sess:

    # =============== problem==============
    init_step = global_step.eval(sess=sess)
    tf.train.start_queue_runners(sess)
    for step in xrange(0, max_step):
        if step % val_interval == 0:
            print "==========val %d=========" %step
            ave_val_loss, ave_val_acc, ave_val_prec, ave_val_rec, ave_val_f1, ave_conf_mat\
                = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for i in range(val_itr):
                np_val_images, np_val_labels = sess.run([val_images, val_labels])
                val_preds, val_loss, val_acc = sess.run([preds, cross_entropy, acc], feed_dict={x: np_val_images, y_: np_val_labels})
                prec, rec, f1, conf_mat = evaluate(np_val_labels, val_preds)
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
        np_train_images, np_train_labels = sess.run([train_images, train_labels])
        _, train_preds, train_loss, train_acc  = sess.run([train_op, preds, cross_entropy, acc], feed_dict={x:np_train_images, y_:np_train_labels})
        if step % train_interval == 0 or step < 10:
            print "==========train %d========" %step
            prec, rec, f1, conf_mat = evaluate(np_train_labels, train_preds)
            print "loss: ", train_loss, "acc: ", train_acc
            print "prec: ", prec
            print "rec: ", rec
            print "conf_mat: "
            print conf_mat

        # if step % 500 == 0:
        #     checkpoint_path = os.path.join(logdir, 'model.ckpt')
        #     sv.saver.save(sess, checkpoint_path, global_step=step)