from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils
import sys


class ResNet(object):
    def __init__(self, num_class, name=None):
        self._num_classes = num_class
        self._name = name
        self.lr = tf.placeholder(tf.float32, name="lr")
        # self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.is_train = tf.constant(False, dtype=tf.bool, name="is_train")
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

    def build_tower(self, images):
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(images, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')
        # x = tf.stop_gradient(x)

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')
        x = tf.stop_gradient(x)

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            # x = tf.reduce_mean(x, [0, 1])
            x = self._fc(x, self._num_classes)

        probs = tf.nn.softmax(x)
        preds = tf.to_int32(tf.argmax(probs, 1))
        print "preds name {}".format(preds.name)
        # return probs
        return tf.stop_gradient(probs)

    def build_new_tower(self, images):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            tf.get_variable_scope().reuse_variables()
            print "reuse: ", tf.get_variable_scope().reuse
            print "scope: ", tf.get_variable_scope().name
            print('Building model')
            # filters = [128, 128, 256, 512, 1024]
            filters = [64, 64, 128, 256, 512]
            kernels = [7, 3, 3, 3, 3]
            strides = [2, 0, 2, 2, 2]
            # conv1
            print('\tBuilding unit: conv1')
            with tf.variable_scope('conv1'):
                x = self._conv(images, kernels[0], filters[0], strides[0])
                x = self._bn(x)
                x = self._relu(x)
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

            # conv2_x
            x = self._residual_block(x, name='conv2_1')
            x = self._residual_block(x, name='conv2_2')

            # conv3_x
            x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
            x = self._residual_block(x, name='conv3_2')

            # conv4_x
            x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
            x = self._residual_block(x, name='conv4_2')
            # x = tf.stop_gradient(x)

            # conv5_x
            x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
            x = self._residual_block(x, name='conv5_2')
            x = tf.stop_gradient(x)


        # with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        #     print('\tBuilding unit: %s' % scope.name)
        # tf.stop_gradient(x)

        print "reuse: ", tf.get_variable_scope().reuse
        print "scope: ", tf.get_variable_scope().name
        with tf.variable_scope('q_logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            # x = self._fc(x, 1024)
            x = self._fc(x, self._num_classes)
            logits = x

        # Probs & preds & acc
        return logits



    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, global_step=None, name=name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
