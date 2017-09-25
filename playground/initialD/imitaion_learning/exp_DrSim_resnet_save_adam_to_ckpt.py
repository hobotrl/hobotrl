import os
import signal
import time
import sys
import traceback
from collections import deque
sys.path.append('../../..')
sys.path.append('..')


import numpy as np

import tensorflow as tf
# from playground.resnet import resnet
import resnet
# Environment

hp = resnet.HParams(batch_size=64,
                            num_gpus=1,
                            num_classes=3,
                            weight_decay=0.001,
                            momentum=0.9,
                            finetune=True)

config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True),
    allow_soft_placement=True,
    log_device_placement=False)

images = tf.placeholder(tf.float32, [None, 224, 224, 3], name="imghh")
y_one_hot = tf.placeholder(tf.float32, [None, 3], name="onehothh")

print "========\n" * 5


with tf.variable_scope("learn"):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    res = resnet.ResNet(hp, global_step, name="train")
    pi = res.build_origin_tower(images)
    q = res.build_new_tower(images)
    cross_entropy = -tf.reduce_mean(tf.to_float(y_one_hot) * tf.log(pi))

with tf.name_scope("optimizers"):
    adam = tf.train.AdamOptimizer(1e-3)
    # opt = adam.minimize(coss_entropy)
    opt1 = adam.minimize(pi)
    opt2 = adam.minimize(q)

glov = tf.global_variables()
# print "global varibles: "
for v in glov:
    print "v.name: ", v.name

print "============\n"*5

variables_to_restore = []
print "variables not to restore"
for v in glov:
    if "optimizers" in v.name or \
        "Adam" in v.name or \
        "q_logits" in v.name or \
        "beta1_power" in v.name or \
        "beta2_power" in v.name:
        # print v.name
        pass
    else:
        variables_to_restore.append(v)

saver = tf.train.Saver(variables_to_restore)
checkpoint = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/DrSim_resnet_rename_pi/model.ckpt"
new_checkpoint = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/DrSim_resnet_rename_pi_q_opt/model.ckpt"
new_saver = tf.train.Saver(glov)


# slot_variables = [adam._beta1_power, adam._beta2_power]
# print "slot variables: ", slot_variables
# all_variables = glov.extend(glov)
# init = tf.variables_initializer(all_variables)

init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init)

    # try to initialize uninitialized vars
    # uninitialized_vars = []
    # for var in tf.all_variables():
    #     try:
    #         sess.run(var)
    #     except tf.errors.FailedPreconditionError:
    #         print "unitilized vars: ", var
    #         uninitialized_vars.append(var)
    # init_new_vars_op = tf.initialize_variables(uninitialized_vars)
    # sess.run(init_new_vars_op)
    print "============\n"*5

    print "restore all variables"
    saver.restore(sess, checkpoint)

    new_saver.save(sess, new_checkpoint, global_step=0)

    # print "unitialized vars after restored"
    # uninitialized_vars = []
    # for var in tf.all_variables():
    #     try:
    #         sess.run(var)
    #     except tf.errors.FailedPreconditionError:
    #         print "unitilized vars: ", var

    # print "global variables: "
    # for v in tf.global_variables():
    #     print v.name


# variables_to_save = []
# for v in glov:
#     if  "beta1_power" in v.name or \
#         "beta2_power" in v.name:
#         # print v.name
#         pass
#     else:
#         variables_to_save.append(v)

