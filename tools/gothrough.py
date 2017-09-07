import tensorflow as tf

# To test concat records file is ok
filename = "/home/pirate03/PycharmProjects/hobotrl/data/records_v1/S5/filter_a3.tfrecords"

i = 0
for example_serialized in tf.python_io.tf_record_iterator(filename):
    i += 1

print("this file is ok and total %d! " %i)