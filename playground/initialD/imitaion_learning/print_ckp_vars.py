import tensorflow as tf

checkpoint_dir = "/home/pirate03/PycharmProjects/hobotrl/playground/initialD/imitaion_learning/fnet_rename"
checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

with tf.Session() as sess:
    var_names = []
    vars = []
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
        var_names.append(var_name)
        var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
        vars.append(var)
    print var_names
    print vars
    print tf.global_variables()

