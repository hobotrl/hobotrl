import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import Image


def filter(data_path = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/S5-2 to S5-4.tfrecords"):
    # data_path = "/home/pirate03/PycharmProjects/resnet-18-tensorflow/val.tfrecords"
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
        action = tf.cast(features["action"], tf.int32)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        train_writer = tf.python_io.TFRecordWriter("/home/pirate03/PycharmProjects/hobotrl/data/records_v1/S5/filter_a3.tfrecords")
        # val_writer = tf.python_io.TFRecordWriter("/home/pirate03/PycharmProjects/hobotrl/data/records_v1/v2_val.tfrecords")

        i = 0
        try:
            while True:
                i += 1
                example, n_eps, n_step, n_action = sess.run([serialized_example, eps, step, action])
                print("i:",i," eps:",n_eps," step:",n_step," action:", n_action)
                if n_action != 3:
                    # if i % 5 == 0:
                    #     val_writer.write(example)
                    # else:
                    #     train_writer.write(example)
                    train_writer.write(example)
        except tf.errors.OutOfRangeError:
            print("Go through all!")
        finally:
            coord.request_stop()
        coord.join(threads)
        train_writer.close()
        # val_writer.close()

def split(data_path = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/S5-2 to S5-4.tfrecords"):
    # data_path = "/home/pirate03/PycharmProjects/resnet-18-tensorflow/val.tfrecords"
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
        action = tf.cast(features["action"], tf.int32)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        a0 = tf.python_io.TFRecordWriter("/home/pirate03/PycharmProjects/hobotrl/data/records_v1/a0.tfrecords")
        a1 = tf.python_io.TFRecordWriter("/home/pirate03/PycharmProjects/hobotrl/data/records_v1/a1.tfrecords")
        a2 = tf.python_io.TFRecordWriter("/home/pirate03/PycharmProjects/hobotrl/data/records_v1/a2.tfrecords")

        i = 0
        try:
            while True:
                i += 1
                example, n_eps, n_step, n_action = sess.run([serialized_example, eps, step, action])
                print("i:",i," eps:",n_eps," step:",n_step," action:", n_action)
                if n_action == 0:
                    a0.write(example)
                elif n_action == 1:
                    a1.write(example)
                elif n_action == 2:
                    a2.write(example)
                else:
                    pass
        except tf.errors.OutOfRangeError:
            print("Go through all!")
        finally:
            coord.request_stop()
        coord.join(threads)
        a0.close()
        a1.close()
        a2.close()


def stat(data_path = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/S5-2 to S5-4.tfrecords"):
    # 'all eps: ', [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]
    # maybe lack records of honda_S5-1.launch
    # 'all actions num: ', [5851, 172, 196, 6174]
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
        action = tf.cast(features["action"], tf.int32)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        i = 0
        all_eps = []
        last_eps = -1
        all_action = [0, 0, 0, 0]

        try:
            while not coord.should_stop():
                while True:
                    i += 1
                    n_eps, n_step, n_action = sess.run([eps, step, action])
                    if n_eps != last_eps:
                        all_eps.append(n_eps)
                        last_eps = n_eps
                    # n_step = sess.run(step)
                    # n_action = sess.run(action)
                    all_action[n_action] += 1
                    print("i:", i)
                    print("eps:", n_eps)
                    print("step:", n_step)
                    print("action:", n_action)
        except Exception, e:
            print("Go through all!")
            print("all eps: ", all_eps)
            print("all actions num: ", all_action)
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == "__main__":
    filter()



