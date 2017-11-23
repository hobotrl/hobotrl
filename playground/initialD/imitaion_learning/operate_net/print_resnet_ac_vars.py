import tensorflow as tf
from pprint import pprint
import hobotrl as hrl
from playground.resnet import resnet_pq


def f_net(inputs):
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
    # saver.restore(sess, checkpoint)
    state = inputs[0]
    print "global varibles: ", tf.global_variables()
    print "========\n"*5
    res = resnet_pq.ResNet(3, name="train")
    pi = res.build_tower(state)
    q = res.build_new_tower(state)

    print "========\n"*5

    # pi = tf.nn.softmax(pi)
    # q = res.build_new_tower(state)
    # print "q type: ", type(q)
    # return {"q":q, "pi": pi}
    return {"pi": pi, "q": q}


state_shape = (256, 256, 9)
global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )
agent = hrl.ActorCritic(
            f_create_net=f_net,
            state_shape=state_shape,
            # ACUpdate arguments
            discount_factor=0.9,
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

checkpoint_dir = "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pq_rename"


with tf.Session() as sess:
    var_names = []
    # vars = []
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
        var_names.append(var_name)
        # var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
        # vars.append(var)
    # pprint(var_names)
    # for name, var in zip(var_names, vars):
    #     print name, ": ", var.shape
    print "checkpoint vars: "
    pprint(var_names)

    all_vars = tf.all_variables()
    print "all vars: "
    pprint(all_vars)

    # print vars
