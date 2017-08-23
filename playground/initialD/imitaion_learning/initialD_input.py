#
# This code is modified from the TensorFlow tutorial below.
#
# TensorFlow Tutorial - Convolutional Neural Networks
#  (https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html)
#
# ==============================================================================

"""Routine for loading the image file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cPickle as pickle
import numpy as np

from tensorflow.python.platform import gfile

# RESNET_MEAN_FPATH = 'ResNet_mean_rgb.pkl'
# with open(RESNET_MEAN_FPATH, 'rb') as fd:
    # resnet_mean = pickle.load(fd).mean(0).mean(0)

# Constants used in the model
RESIZE_SIZE = 256
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


def read_input_file(txt_fpath, dataset_root, shuffle=False):
    """Reads and parses examples from AwA data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      list_fpath: Path to a txt file containing subpath of input image and labels
        line-by-line
      dataset_root: Path to the root of the dataset images.

    Returns:
      An object representing a single example, with the following fields:
        path: a scalar string Tensor of the path to the image file.
        labels: an int32 Tensor with the 64 attributes(0/1)
        image: a [height, width, depth(BGR)] float32 Tensor with the image data
    """

    class DataRecord(object):
        pass
    result = DataRecord()

    # Read a line from the file(list_fname)
    filename_queue = tf.train.string_input_producer([txt_fpath], shuffle=shuffle)
    text_reader = tf.TextLineReader()
    _, value = text_reader.read(filename_queue)

    # Parse the line -> subpath, label
    record_default = [[''], [0]]
    parsed_entries = tf.decode_csv(value, record_default, field_delim=' ')
    result.labels = tf.cast(parsed_entries[1], tf.int32)

    # Read image from the filepath
    # image_path = os.path.join(dataset_root, parsed_entries[0])
    dataset_root_t = tf.constant(dataset_root)
    result.image_path = dataset_root_t + parsed_entries[0] # String tensors can be concatenated by add operator
    raw_jpeg = tf.read_file(result.image_path)
    result.image = tf.image.decode_jpeg(raw_jpeg, channels=3)

    return result


def resize_image(input_image, random_aspect=False):
    # Resize image so that the shorter side is 256
    height_orig = tf.shape(input_image)[0]
    width_orig = tf.shape(input_image)[1]
    ratio_flag = tf.greater(height_orig, width_orig)  # True if height > width
    if random_aspect:
        aspect_ratio = tf.random_uniform([], minval=0.875, maxval=1.2, dtype=tf.float64)
        height = tf.where(ratio_flag, tf.cast(RESIZE_SIZE*height_orig/width_orig*aspect_ratio, tf.int32), RESIZE_SIZE)
        width = tf.where(ratio_flag, RESIZE_SIZE, tf.cast(RESIZE_SIZE*width_orig/height_orig*aspect_ratio, tf.int32))
    else:
        height = tf.where(ratio_flag, tf.cast(RESIZE_SIZE*height_orig/width_orig, tf.int32), RESIZE_SIZE)
        width = tf.where(ratio_flag, RESIZE_SIZE, tf.cast(RESIZE_SIZE*width_orig/height_orig, tf.int32))
    image = tf.image.resize_images(input_image, [height, width])
    return image


def random_sized_crop(input_image):
    # Input image -> crop with random size and random aspect ratio
    height_orig = tf.cast(tf.shape(input_image)[0], tf.float64)
    width_orig = tf.cast(tf.shape(input_image)[1], tf.float64)

    aspect_ratio = tf.random_uniform([], minval=0.75, maxval=1.33, dtype=tf.float64)
    height_max = tf.minimum(height_orig, width_orig*aspect_ratio)
    height_crop = tf.random_uniform([], minval=tf.minimum(height_max, tf.maximum(0.5*height_orig, 0.5*height_max))
                                    , maxval=height_max, dtype=tf.float64)
    width_crop = height_crop / aspect_ratio
    height_crop = tf.cast(height_crop, tf.int32)
    width_crop = tf.cast(width_crop, tf.int32)

    crop = tf.random_crop(input_image, [height_crop, width_crop, 3])

    # Resize to 224x224
    image = tf.image.resize_images(crop, [IMAGE_HEIGHT, IMAGE_WIDTH])

    return image


def lighting(input_image):
    # Lighting noise (AlexNet-style PCA-based noise) from torch code
    # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua
    alphastd = 0.1
    eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    eigvec = np.array([[-0.5675,  0.7192,  0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

    alpha = tf.random_normal([3, 1], mean=0.0, stddev=alphastd)
    rgb = alpha * (eigval.reshape([3, 1]) * eigvec)
    image = input_image + tf.reduce_sum(rgb, axis=0)

    return image


def preprocess_image(input_image):
    # Preprocess the image: resize -> mean subtract -> channel swap (-> transpose X -> scale X)
    # image = tf.cast(input_image, tf.float32)
    # image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    # image_R, image_G, image_B = tf.split(2, 3, image)

    # 1) Subtract channel mean
    # blue_mean = 103.062624
    # green_mean = 115.902883
    # red_mean = 123.151631
    # image = tf.concat(2, [image_B - blue_mean, image_G - green_mean, image_R - red_mean], name="centered_bgr")

    # 2) Subtract per-pixel mean(the model have to 224 x 224 size input)
    # image = tf.concat(2, [image_B, image_G, image_R]) - resnet_mean
    # image = image - resnet_mean

    # image = tf.concat(2, [image_R, image_G, image_B]) # BGR -> RGB
    # imagenet_mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    # image = image - imagenet_mean # [224, 224, 3] - [3] (Subtract with broadcasting)
    # image = tf.transpose(image, [2, 0, 1]) # No transpose
    # No scaling

    # NEW: Computed from random subset of ImageNet training images
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (input_image - imagenet_mean) / imagenet_std

    return image


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle=True, num_threads=60):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of [NUM_ATTRS] of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Attribute labels. 2D tensor of [batch_size, NUM_ATTRS] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.

    if not shuffle:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 10 * batch_size)
    else:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 20 * batch_size,
            min_after_dequeue=min_queue_examples)

    return images, label_batch


def distorted_inputs(filename, batch_size, shuffle=True, num_threads=10):
    """Construct distorted input for IMAGENET training using the Reader ops.

    Args:
      data_class: string, indicating if one should use the 'train' or 'eval' or 'test' data set.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'eps': tf.FixedLenFeature([], tf.int64),
                                           'step': tf.FixedLenFeature([], tf.int64),
                                           'state': tf.FixedLenFeature([], tf.string),
                                           'action' : tf.FixedLenFeature([], tf.int64),
                                           'reward': tf.FixedLenFeature([], tf.float32)
                                       })

    state = tf.decode_raw(features['state'], tf.uint8)
    state = tf.reshape(state, [640, 640, 3])
    print("state name : ", state.name)
    state = tf.image.resize_images(state, [IMAGE_HEIGHT, IMAGE_WIDTH])
    state = tf.image.convert_image_dtype(state, tf.float32)
    # state = tf.cast(state, tf.float32) * (1. / 255) - 0.5
    # state = tf.cast(state, tf.float32) * (1. / 255)
    state = preprocess_image(state)
    action = tf.cast(features['action'], tf.int32)
    min_queue_examples = batch_size * 10
    state_batch, action_batch = _generate_image_and_label_batch(
        state, action, min_queue_examples, batch_size, shuffle=shuffle, num_threads=num_threads)
    print("state_batch name: ", state_batch.name)
    return [state_batch], [action_batch]


def inputs(filename, batch_size, shuffle=False, num_threads=10):
    """Construct input for IMAGENET evaluation using the Reader ops.

    Args:
      data_class: string, indicating if one should use the 'train' or 'eval' or 'test' data set.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'eps': tf.FixedLenFeature([], tf.int64),
                                           'step': tf.FixedLenFeature([], tf.int64),
                                           'state': tf.FixedLenFeature([], tf.string),
                                           'action': tf.FixedLenFeature([], tf.int64),
                                           'reward': tf.FixedLenFeature([], tf.float32)
                                       })

    state = tf.decode_raw(features['state'], tf.uint8)
    state = tf.reshape(state, [640, 640, 3])
    state = tf.image.resize_images(state, [IMAGE_HEIGHT, IMAGE_WIDTH])
    state = tf.image.convert_image_dtype(state, tf.float32)
    state = preprocess_image(state)
    action = tf.cast(features['action'], tf.int32)
    min_queue_examples = batch_size * 10
    state_batch, action_batch = _generate_image_and_label_batch(
        state, action, min_queue_examples, batch_size, shuffle=shuffle, num_threads=num_threads)
    return [state_batch], [action_batch]


def init_replay_buffer(filename, sess, buffer_size=2000, batch_size=200):
    replay_buffer = []
    imgs, acts = distorted_inputs(filename, batch_size, True, num_threads=6)

    for i in range(buffer_size / batch_size):
        np_imgs, np_acts = sess.run([imgs, acts])
        replay_buffer.append([np_imgs[j], np_acts[j]] for j in range(batch_size))

    return replay_buffer
