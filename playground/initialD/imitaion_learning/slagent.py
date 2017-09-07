# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

import logging
import tensorflow as tf
import numpy as np
import hobotrl as hrl
import hobotrl.network as network
import hobotrl.sampling as sampling
# from hobotrl.tf_dependent.base import BaseDeepAgent
from hobotrl.core import BaseAgent
from hobotrl.playback import MapPlayback
import hobotrl.target_estimate as target_estimate
from playground.initialD.imitaion_learning import resnet

checkpoint = ""

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100000)
print('Load checkpoint %s' % checkpoint)
saver.restore(sess, checkpoint)
global_step = tf.Variable(0, trainable=False, name='global_step')
init_step = global_step.eval(session=sess)
batchsize = 1
hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=FLAGS.num_gpus,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)
network_train = resnet.ResNet(hp, train_images, train_labels, global_step, name="train")
network_val = resnet.ResNet(hp, val_images, val_labels, global_step, name="val", reuse_weights=True)


class SLAgent(BaseAgent):
    """
    Using deterministic policy
    """
    def __init__(self,
                 policy, state_shape, *args, **kwargs):

        super(BaseAgent, self).__init__(*args, **kwargs)
        self.poliy = policy

    def step(self):
        pass

    def act(self, state, **kwargs):
        return self.policy(np.asarray([state]))[0]



# class SLAgent(BaseAgent):
#     """
#     Using deterministic policy
#     """
#     def __init__(self,
#                  f_policy, state_shape, *args, **kwargs):
#         kwargs.update({
#             "f_policy": f_policy,
#             "state_shape": state_shape,
#         })
#         super(SLAgent, self).__init__(*args, **kwargs)
#         self.poliy = network.NetworkFunction(self.network)
#
#     def init_network(self, policy_net, state_shape, *args, **kwargs):
#         input_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="input_state")
#         return network.Network([input_state], policy_net, var_scope="learn")
#
#     def step(self):
#         pass
#
#     def act(self, state, **kwargs):
#         return self.policy(np.asarray([state]))[0]



