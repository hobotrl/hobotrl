# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops


@ops.RegisterGradient("ATanhGrad")
def atanh_grad(op, grad):
    x = op.inputs[0]
    y = op.outputs[0]
    org_grad = gen_math_ops._tanh_grad(y, grad)
    # return [gen_math_ops._abs(gen_math_ops.sign(y) - gen_math_ops.sign(grad)) / 2 * org_grad +
    #         gen_math_ops._abs(gen_math_ops.sign(y) + gen_math_ops.sign(grad)) / 2 * grad]
    return [gen_math_ops._abs(gen_math_ops.sign(y) - gen_math_ops.sign(grad)) * org_grad +
            gen_math_ops._abs(gen_math_ops.sign(y) + gen_math_ops.sign(grad)) / 2 * grad]


def atanh(x, name=None):
    # tf.RegisterGradient("AtanhGrad")(atanh_grad)
    with ops.name_scope(name, "ATanh", [x]) as name:
        g = tf.get_default_graph()
        with g.gradient_override_map({"Tanh": "ATanhGrad"}):
            return tf.tanh(x)
