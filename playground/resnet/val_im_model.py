#!/usr/bin/env python

import time

import numpy as np
import resnet
import tensorflow as tf
from playground.initialD.imitaion_learning.sl.evaluate import evaluate
from playground.initialD.imitaion_learning.process_data.stack_imgs import read_eps_imgs_acts, stack_one_eps
from playground.initialD.imitaion_learning.process_data.stack_imgs import stack_obj_eps
from playground.initialD.imitaion_learning.split_stack_infos import concat_imgs_acts

# Dataset Configuration
tf.app.flags.DEFINE_string('val_dir', '/home/pirate03/hobotrl_data/playground/initialD/exp/record_rule_scenes_obj80_vec_rewards_docker005_no_early_stopping_all_green/valid', """Path to initialD the test dataset""")
tf.app.flags.DEFINE_integer('num_classes', 3, """Number of classes in the dataset.""")
# Training Configuration
tf.app.flags.DEFINE_string('log_dir', './resnet/val_ckp', """Directory where to write log and checkpoint.""")

FLAGS = tf.app.flags.FLAGS
val_dir = FLAGS.val_dir
log_dir = FLAGS.log_dir

hp = resnet.HParams(batch_size=64,
                    num_gpus=1,
                    num_classes=3,
                    weight_decay=0.001,
                    momentum=0.9,
                    finetune=True)
global_step = tf.Variable(0, trainable=False, name='global_step')
network_val = resnet.ResNet(hp, global_step, name="val")
network_val.build_model()
stack_num = 3
state_shape = (256, 256, 3*stack_num)
labels = [0, 1, 2]

graph = tf.get_default_graph()

init_op = tf.global_variables_initializer()

sv = tf.train.Supervisor(graph=graph,
                        global_step=global_step,
                        init_op=init_op,
                        summary_op=None,
                        summary_writer=None,
                        logdir=log_dir,)
                        # save_summaries_secs=0)

config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
            allow_soft_placement=True,
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
    eps_names = sorted(os.listdir(val_dir))
    # print eps_names
    for eps_name in eps_names:
        eps_dir = val_dir + "/" + eps_name
        img_names = sorted(os.listdir(eps_dir))[1:]
        # start_time = int(img_names[0].split('.')[0].split('_')[0])-1
        # eps_stat_txt = open(eps_dir+"/0000.txt", 'r').readlines()
        # eps_stat_txt = eps_stat_txt[start_time:]
        # new_eps_stat_txt = []
        eps_imgs, eps_acts = read_eps_imgs_acts(eps_dir)
        eps_stack_info = stack_one_eps(eps_imgs, eps_acts, stack_num)
        eps_labels = []
        eps_preds = []
        val_imgs, val_labels = concat_imgs_acts(eps_stack_info)
        eps_infer_time = 0.0
        for i in range(len(val_imgs)):
            val_img = val_imgs[i]
            val_label = val_labels[i]
            start = time.time()
            loss_value, acc_value, val_preds = sess.run([network_val.loss, network_val.acc, network_val.preds],
                                                    feed_dict={network_val._images: np.array([val_img]),
                                                               network_val._labels: np.array([val_label]),
                                                               network_val.is_train: False})
            eps_infer_time += time.time() - start
            eps_labels.append(val_label)
            eps_preds.append(val_preds[0])

        for i, val_pred in enumerate(eps_preds):
            new_name = img_names[i+stack_num].split('.')[0]+"_"+str(val_pred)+".jpg"
            os.rename(eps_dir+"/"+img_names[i+stack_num], eps_dir+"/"+new_name)
            # new_eps_stat_txt.append(eps_stat_txt[i].split('\n')[0]+","+str(val_pred)+"\n")

        # f = open(eps_dir+"/0001.txt", "w")
        # for line in new_eps_stat_txt:
        #     f.write(line)
        # f.close()
        prec, rec, f1, conf_mat = evaluate(np.array(eps_labels), np.array(eps_preds), labels)
        obj_preds.extend(eps_preds)
        obj_labels.extend(eps_labels)
        print "eps_name: ", eps_name, "prec: ", prec, "rec: ", rec, "f1: ", f1
        print "conf_mat: \n", conf_mat
        print "infer time: ", eps_infer_time / len(val_imgs)

    stat_prec, stat_rec, stat_f1, stat_conf_mat = evaluate(np.array(obj_labels), np.array(obj_preds), labels)
    print "stat result"
    print "prec: ", prec, "rec: ", rec, "f1: ", f1
    print "conf_mat: \n", conf_mat





