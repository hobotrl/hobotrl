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
import playground.initialD.imitaion_learning.resnet
import hobotrl as hrl


hp = playground.initialD.imitaion_learning.resnet.HParams(batch_size=64,
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

def f_net(inputs):
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    # saver.restore(sess, checkpoint)
    state = inputs[0]
    print "global varibles: ", tf.global_variables()
    print "========\n"*5
    res = playground.initialD.imitaion_learning.resnet.ResNet(hp, global_step, name="train")
    pi = res.build_origin_tower(state)
    q = res.build_new_tower(state)

    print "========\n"*5

    # pi = tf.nn.softmax(pi)
    # q = res.build_new_tower(state)
    # print "q type: ", type(q)
    # return {"q":q, "pi": pi}
    return {"pi":pi, "q":q}

global_step = tf.get_variable(
            'learn/global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

agent = hrl.ActorCritic(
            f_create_net=f_net,
            state_shape=(224, 224, 3),
            # ACUpdate arguments
            discount_factor=0.7,
            entropy=hrl.utils.CappedLinear(1e6, 1e-2, 1e-2),
            target_estimator=None,
            max_advantage=100.0,
            # optimizer arguments
            network_optmizer=hrl.network.LocalOptimizer(tf.train.AdamOptimizer(1e-3), grad_clip=10.0),
            max_gradient=10.0,
            # sampler arguments
            sampler=None,
            batch_size=8,
            global_step=global_step,
        )

glov = tf.global_variables()
# print "global varibles: "
for v in glov:
    print "v.name: ", v.name

print "============\n"*5

variables_to_restore = []
print "variables not to restore"
for v in glov:
    # if "optimizers" in v.name or \
    #     "Adam" in v.name or \
    #     "q_logits" in v.name or \
    #     "beta1_power" in v.name or \
    #     "beta2_power" in v.name:
    #     # print v.name
    #     pass
    if "Adam" in v.name:
        pass
    else:
        variables_to_restore.append(v)

saver = tf.train.Saver(variables_to_restore)
checkpoint = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/DrSim_resnet_rename_pi_q_opt_learn_q/model.ckpt-13761"
new_checkpoint = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/DrSim_resnet_rename_pi_q_opt_add_logitsadam/model.ckpt"
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

