import tensorflow as tf

gpu_fraction = 0.5
config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction),
            # allow_soft_placement=False,
            allow_soft_placement=True)
sess = tf.Session(config=config)
checkpoint = "/home/pirate03/PycharmProjects/resnet-18-tensorflow/log3_2/model.ckpt-10000"
saver = tf.train.import_meta_graph(
            '/home/pirate03/PycharmProjects/resnet-18-tensorflow/log3_2/model.ckpt-10000.meta',
            clear_devices=True)
saver.restore(sess, checkpoint)
graph = tf.get_default_graph()
train_op = graph.get_operation_by_name("group_deps")
sess.run(train_op)

sess.close()