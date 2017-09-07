import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import Image

# data_path = "/home/pirate03/PycharmProjects/resnet-18-tensorflow/val.tfrecords"
# data_path = "/home/pirate03/PycharmProjects/hobotrl/tools/v2_train_valid_writer.tfrecords"
data_path = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/S5-2 to S5-4.tfrecords"
# with tf.Session() as sess:
#     feature = {"eps": tf.FixedLenFeature([], tf.int64),
#                "step": tf.FixedLenFeature([], tf.int64),
#                "state": tf.FixedLenFeature([], tf.string),
#                "action": tf.FixedLenFeature([], tf.int64)}
#     filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example, features=feature)
#     eps = tf.cast(features["eps"], tf.int32)
#     step = tf.cast(features["step"], tf.int32)
#     state = tf.decode_raw(features["state"], tf.uint8)
#     action = tf.cast(features["action"], tf.int32)
#     state = tf.reshape(state, [640, 640, 3])
#     epses, steps, states, actions = \
#         tf.train.shuffle_batch([eps, step, state, action], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
#
#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess.run(init_op)
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for batch_index in range(5):
#         eps, step, state, action = sess.run([epses, steps, states, actions])
#         state = state.astype(np.uint8)
#         for j in range(6):
#             plt.subplot(2, 3, j+1)
#             plt.imshow(state[j,...])
#             plt.title(str(eps[j,...])+"_"+str(step[j,...]))
#         plt.show()
#
#     coord.request_stop()
#
#     coord.join(threads)
#     sess.close()


with tf.Session() as sess:
    feature = {"eps": tf.FixedLenFeature([], tf.int64),
               "step": tf.FixedLenFeature([], tf.int64),
               "state": tf.FixedLenFeature([], tf.string),
               "action": tf.FixedLenFeature([], tf.int64),
               "reward": tf.FixedLenFeature([], tf.float32)}
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature)
    eps = tf.cast(features["eps"], tf.int32)
    step = tf.cast(features["step"], tf.int32)
    state = tf.decode_raw(features["state"], tf.uint8)
    action = tf.cast(features["action"], tf.int32)
    reward = tf.cast(features["reward"], tf.float32)
    state = tf.reshape(state, [640, 640, 3])
    # epses, steps, states, actions =
    #     tf.train.shuffle_batch([eps, step, state, action], batch_size=10, capacity=30, num_threads=1,
    #                            min_after_dequeue=10)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    i = 0
    try:
        while True:
            print("i:",i)
            img = sess.run(state)
            img = img.astype(np.uint8)
            cv2.imshow("test", img)
            cv2.waitKey(0)
            print(i)
            e, t, a, r = sess.run([eps, step, action, reward])
            i += 1
            print("e:",e)
            print("t:",t)
            print("a:",a)
            print("r:", r)
    except tf.errors.OutOfRangeError:
        print("Go through all!")
    finally:
        coord.request_stop()

    coord.join(threads)