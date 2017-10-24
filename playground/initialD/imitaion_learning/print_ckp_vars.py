import tensorflow as tf
from pprint import pprint
checkpoint_dir = "/home/pirate03/PycharmProjects/hobotrl/playground/resnet/resnet_pq"
checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

with tf.Session() as sess:
    var_names = []
    vars = []
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
        var_names.append(var_name)
        var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
        vars.append(var)
    pprint(var_names)

    for name, var in zip(var_names, vars):
        print name, ": ", var.shape
    # print vars

