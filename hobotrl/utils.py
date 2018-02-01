# -*- coding: utf-8 -*-

"""Main building blocks for RL agent.
Guys who do the heavy-lifting lies here. Supposed to be friendly to
both mixin-style and non-... usages.

1. TabularQFunc
2. EpsilonGreedyPolicy
3. ExperienceReplay
"""

import re
import math
import numpy as np
from numpy import max
from numpy.random import rand, randint
import tensorflow.contrib.layers as layers
import tensorflow as tf





class EpsilonGreedyPolicy(object):
    """Epsilon greedy policy
    This policy superimpose a random policy onto the greedy policy with a small
    probability epsilon.

    Assume super-class already initialized a discrete and index-able action
    space. And assume a action-value func. mixin already implements the
    "get_value()" method for value retrieval.
    """
    def __init__(self, actions, f_get_value, epsilon, tol=1e-10, **kwargs):
        """Initialization

        Parameters
        ----------
        actions :
        f_get_value :
        epsilon : probability of choosing random action.
        tol     : a small tolerance for equality tests.
        """
        self.__ACTIONS = actions
        self.__get_value = f_get_value
        self.__EPSILON = epsilon
        self.__TOL = tol

    def act_single_(self, state, **kwargs):
        """Epsilon greedy action selection.
        Choose greedy action with 1-epsilon probability and random action with
        epsilon probability. Ties are broken randomly for greedy actions.
        """
        exploration_off = kwargs['exploration_off'] \
            if 'exploration_off' in kwargs else False
        if state is None:
            idx_action = randint(0, len(self.__ACTIONS))
        elif not exploration_off and rand() < self.__EPSILON:
            idx_action = randint(0, len(self.__ACTIONS))
        else:
            # Follow greedy policy with 1-epsilon prob.
            # break tie randomly
            q_vals = np.asarray(
                self.__get_value(state=state, **kwargs)
            ).flatten()
            max_q_val = max(q_vals)
            idx_best_actions = [
                i for i in range(len(q_vals))
                if (q_vals[i] - max_q_val)**2 < self.__TOL
            ]
            idx_action = idx_best_actions[randint(0, len(idx_best_actions))]
        return self.__ACTIONS[idx_action]


class Network(object):
    """
    :deprecated by hobotrl.network.network.Utils
    """
    @staticmethod
    def layer_fcs(input_var, shape, out_count, activation_hidden=tf.nn.relu, activation_out=None, l2=0.0001,
                  var_scope=""):
        variables = []
        ops = []
        with tf.variable_scope(var_scope):
            for i in range(len(shape)):
                hidden_count = shape[i]
                out = layers.fully_connected(inputs=input_var, num_outputs=hidden_count,
                                             activation_fn=activation_hidden,
                                             weights_initializer=layers.xavier_initializer(),
                                             biases_initializer=layers.xavier_initializer(),
                                             weights_regularizer=layers.l2_regularizer(l2),
                                             biases_regularizer=layers.l2_regularizer(l2),
                                             scope="hidden_%d" % i)
                input_var = out

            # output
            with tf.variable_scope("out"):
                out = layers.fully_connected(inputs=input_var, num_outputs=out_count,
                                             activation_fn=activation_out,
                                             weights_initializer=layers.xavier_initializer(),
                                             biases_initializer=layers.xavier_initializer(),
                                             weights_regularizer=layers.l2_regularizer(l2),
                                             biases_regularizer=layers.l2_regularizer(l2),
                                             scope="out")
        return out

    @staticmethod
    def conv2d(input_var, h, w, out_channel, strides=[1, 1], padding="SAME",
               activation=tf.nn.relu, l2=1e-4, var_scope=""):
        with tf.variable_scope(var_scope):
            out = tf.layers.conv2d(inputs=input_var, filters=out_channel, kernel_size=[w, h],
                                   strides=strides, padding=padding, activation=activation,
                                   use_bias=True, kernel_initializer=layers.xavier_initializer(),
                                   # bias_initializer=layers.xavier_initializer(),
                                   kernel_regularizer=layers.l2_regularizer(l2),
                                   bias_regularizer=layers.l2_regularizer(l2))
        return out

    @staticmethod
    def conv2ds(input_var, shape=[(64, 4, 1)], out_flatten=True, padding="SAME",
                activation=tf.nn.relu, l2=1e-4, var_scope=""):
        out = input_var
        with tf.variable_scope(var_scope):
            for i in range(len(shape)):
                s = shape[i]
                filter_n, kernel_n, strides_n = s
                out = Network.conv2d(out, h=kernel_n, w=kernel_n, out_channel=filter_n,
                                     strides=[strides_n, strides_n], padding=padding,
                                     activation=activation, l2=l2, var_scope="conv%d" % i)
        if out_flatten:
            out = tf.contrib.layers.flatten(out)
        return out

    @staticmethod
    def clipped_square(value, clip=1.0):
        abs_value = tf.abs(value)
        quadratic = tf.minimum(abs_value, clip)
        linear = abs_value - quadratic
        return 0.5 * tf.square(quadratic) + clip * linear

    @staticmethod
    def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
        """Minimized `objective` using `optimizer` w.r.t. variables in
        `var_list` while ensure the norm of the gradients for each
        variable is clipped to `clip_val`
        """
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients), gradients


class NP(object):
    @staticmethod
    def one_hot(array, num):
        """Construct a one-hot matrix from an array.
        [0, 2, 1] -> [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

        :param array: N by 0 vector.
        :param num: number of categories.
        :return:
        """
        oh = np.zeros(shape=list(array.shape)+[num])
        oh[np.arange(array.size), array] = 1
        return oh


def escape_path(string):
    return re.sub("[^0-9a-zA-Z\\-_\\.]", "_", string)


class hashable_list(list):
    @staticmethod
    def __new__(S, *args, **kwargs):
        """ T.__new__(S, ...) -> a new object with type S, a subtype of T """
        return list.__new__(S, [])

    def __init__(self, another=None):
        super(hashable_list, self).__init__()
        if another is not None:
            for o in another:
                self.append(o)

    def __hash__(self):
        h = 0
        if len(self) == 0:
            return h
        for o in self:
            h += hash(o)
        return h

    def __eq__(self, other):
        if other is None or len(self) != len(other):
            return False
        for i in range(len(other)):
            if not self[i] == other[i]:
                return False
        return True


class FloatParam(float):
    """
    float(x) -> floating point number

    Convert a string or number to a floating point number, if possible.
    """

    def __init__(self, x):
        super(FloatParam, self).__init__(x)
        self._value = x

    def set(self, v):
        """
        set value of this parameter
        :param v:
        :return:
        """
        if not isinstance(v, float):
            v = float(v)
        self._value = v

    @property
    def value(self):
        return self._value

    def as_integer_ratio(self):
        """
        float.as_integer_ratio() -> (int, int)

        Return a pair of integers, whose ratio is exactly equal to the original
        float and with a positive denominator.
        Raise OverflowError on infinities and a ValueError on NaNs.

        >>> (10.0).as_integer_ratio()
        (10, 1)
        >>> (0.0).as_integer_ratio()
        (0, 1)
        >>> (-.25).as_integer_ratio()
        (-1, 4)
        """
        return self.value.as_integer_ratio()

    def conjugate(self, *args, **kwargs):  # real signature unknown
        """ Return self, the complex conjugate of any float. """
        return self.value.conjugate(*args, **kwargs)

    def hex(self):
        """
        float.hex() -> string

        Return a hexadecimal representation of a floating-point number.
        >>> (-0.1).hex()
        '-0x1.999999999999ap-4'
        >>> 3.14159.hex()
        '0x1.921f9f01b866ep+1'
        """
        return self.value.hex()

    def is_integer(self, *args, **kwargs):  # real signature unknown
        """ Return True if the float is an integer. """
        return self.value.is_integer(*args, **kwargs)

    def __abs__(self):
        """ x.__abs__() <==> abs(x) """
        return self.value.__abs__()

    def __add__(self, y):
        """ x.__add__(y) <==> x+y """
        return self.value.__add__(y)

    def __coerce__(self, y):
        """ x.__coerce__(y) <==> coerce(x, y) """
        return self.value.__coerce__(y)

    def __divmod__(self, y):
        """ x.__divmod__(y) <==> divmod(x, y) """
        return self.value.__divmod__(y)

    def __div__(self, y):
        """ x.__div__(y) <==> x/y """
        return self.value.__div__(y)

    def __eq__(self, y):
        """ x.__eq__(y) <==> x==y """
        return self.value.__eq__(y)

    def __float__(self):
        """ x.__float__() <==> float(x) """
        return self.value.__float__()

    def __floordiv__(self, y):
        """ x.__floordiv__(y) <==> x//y """
        return self.value.__floordiv__(y)

    def __format__(self, format_spec):
        """
        float.__format__(format_spec) -> string

        Formats the float according to format_spec.
        """
        return self.value.__format__(format_spec)

    def __getformat__(self, typestr):
        """
        float.__getformat__(typestr) -> string

        You probably don't want to use this function.  It exists mainly to be
        used in Python's test suite.

        typestr must be 'double' or 'float'.  This function returns whichever of
        'unknown', 'IEEE, big-endian' or 'IEEE, little-endian' best describes the
        format of floating point numbers used by the C type named by typestr.
        """
        return self.value.__getformat__(typestr)

    def __getnewargs__(self, *args, **kwargs):  # real signature unknown
        return self.value.__getnewargs__(*args, **kwargs)

    def __ge__(self, y):
        """ x.__ge__(y) <==> x>=y """
        return self.value.__ge__(y)

    def __gt__(self, y):
        """ x.__gt__(y) <==> x>y """
        return self.value.__gt__(y)

    def __hash__(self):
        """ x.__hash__() <==> hash(x) """
        return self.value.__hash__()

    def __int__(self):
        """ x.__int__() <==> int(x) """
        return self.value.__int__()

    def __le__(self, y):
        """ x.__le__(y) <==> x<=y """
        return self.value.__le__(y)

    def __long__(self):
        """ x.__long__() <==> long(x) """
        return self.value.__long__()

    def __lt__(self, y):
        """ x.__lt__(y) <==> x<y """
        return self.value.__lt__(y)

    def __mod__(self, y):
        """ x.__mod__(y) <==> x%y """
        return self.value.__mod__(y)

    def __mul__(self, y):
        """ x.__mul__(y) <==> x*y """
        return self.value.__mul__(y)

    def __neg__(self):
        """ x.__neg__() <==> -x """
        return self.value.__neg__()

    def __ne__(self, y):
        """ x.__ne__(y) <==> x!=y """
        return self.value.__ne__(y)

    def __nonzero__(self):
        """ x.__nonzero__() <==> x != 0 """
        return self.value.__nonzero__()

    def __pos__(self):
        """ x.__pos__() <==> +x """
        return self.value.__pos__()

    def __pow__(self, y, z=None):
        """ x.__pow__(y[, z]) <==> pow(x, y[, z]) """
        return self.value.__pow__(y, z)

    def __radd__(self, y):
        """ x.__radd__(y) <==> y+x """
        return self.value.__radd__(y)

    def __rdivmod__(self, y):
        """ x.__rdivmod__(y) <==> divmod(y, x) """
        return self.value.__rdivmod__(y)

    def __rdiv__(self, y):
        """ x.__rdiv__(y) <==> y/x """
        return self.value.__rdiv__(y)

    def __repr__(self):
        """ x.__repr__() <==> repr(x) """
        return self.value.__repr__()

    def __rfloordiv__(self, y):
        """ x.__rfloordiv__(y) <==> y//x """
        return self.value.__rfloordiv__(y)

    def __rmod__(self, y):
        """ x.__rmod__(y) <==> y%x """
        return self.value.__rmod__(y)

    def __rmul__(self, y):
        """ x.__rmul__(y) <==> y*x """
        return self.value.__rmul__(y)

    def __rpow__(self, x, z=None):
        """ y.__rpow__(x[, z]) <==> pow(x, y[, z]) """
        return self.value.__rpow__(x, z)

    def __rsub__(self, y):
        """ x.__rsub__(y) <==> y-x """
        return self.value.__rsub__(y)

    def __rtruediv__(self, y):
        """ x.__rtruediv__(y) <==> y/x """
        return self.value.__rtruediv__(y)

    def __setformat__(self, typestr, fmt):
        """
        float.__setformat__(typestr, fmt) -> None

        You probably don't want to use this function.  It exists mainly to be
        used in Python's test suite.

        typestr must be 'double' or 'float'.  fmt must be one of 'unknown',
        'IEEE, big-endian' or 'IEEE, little-endian', and in addition can only be
        one of the latter two if it appears to match the underlying C reality.

        Override the automatic determination of C-level floating point type.
        This affects how floats are converted to and from binary strings.
        """
        return self.value.__setformat__(typestr, fmt)

    def __str__(self):
        """ x.__str__() <==> str(x) """
        return self.value.__str__()

    def __sub__(self, y):
        """ x.__sub__(y) <==> x-y """
        return self.value.__sub__(y)

    def __truediv__(self, y):
        """ x.__truediv__(y) <==> x/y """
        return self.value.__truediv__(y)

    def __trunc__(self, *args, **kwargs):  # real signature unknown
        """ Return the Integral closest to x between 0 and x. """
        return self.value.__trunc__(*args, **kwargs)


class IntHandle(object):
    def __init__(self, n=0):
        super(IntHandle, self).__init__()
        self._n = n

    def value(self):
        return self._n


class Stepper(IntHandle):
    def __init__(self, n=0):
        super(Stepper, self).__init__(n)

    def step(self):
        self._n += 1

    def set(self, n):
        self._n = n

    def get(self):
        return self._n


def clone_param(param):
    if isinstance(param, ScheduledParam):
        return param.clone()
    if isinstance(param, tuple):
        return tuple([clone_param(p) for p in param])
    if isinstance(param, list):
        return [clone_param(p) for p in param]
    if isinstance(param, dict):
        return dict([(k, clone_param(param[k])) for k in param])
    return param


def clone_params(*params, **paramsk):
    if len(params) > 0:
        return clone_param(params[0]) if len(params) == 1 else clone_param(params)
    return clone_param(paramsk)


class ScheduledParam(FloatParam):
    """
    float parameter that changes according to a schedule.
    useful in hyperparameter annealing.
    """

    @staticmethod
    def __new__(S, *args, **kwargs):
        """ T.__new__(S, ...) -> a new object with type S, a subtype of T """
        return float.__new__(S, 0)

    def __init__(self, schedule, int_handle=None, **schedule_params):
        """
        :param schedule: function(n) -> value
        :param int_handle: IntHandle object
        :type int_handle: IntHandle
        """
        self._schedule = schedule
        self._n = int_handle
        self._schedule_params = schedule_params
        x = schedule(0)
        super(ScheduledParam, self).__init__(x)

    def clone(self):
        return ScheduledParam(self._schedule, self._n, **self._schedule_params)

    @property
    def value(self):
        if self._schedule is not None and self._n is not None:
            self._value = self._schedule(self._n.value())
        return super(ScheduledParam, self).value

    def __str__(self):
        value_str = "" if self._n is None else "-[][%d]%f" % (self._n.value(), self._value)
        param_str = "-".join(["{}{:.2e}".format(k, self._schedule_params[k]) for k in self._schedule_params])
        return param_str + value_str

    def __repr__(self):
        return self.__str__()

    def set_int_handle(self, int_handle):
        self._n = int_handle


class ScheduledParamCollector(object):
    def __init__(self, *args, **kwargs):
        super(ScheduledParamCollector, self).__init__()
        self._params = {}
        self._numeric_params = {}
        self.max_depth = 10
        self.max_param_num = 128  # no algorithm should expose more than 128 hyperparameters!

        self.schedule_params("hyperparams", 0, *args, **kwargs)

    def schedule_params(self, prefix,  _spc_depth, *args, **kwargs):
        if _spc_depth >= self.max_depth or len(self._params) >= self.max_param_num:
            return
        for i in range(len(args)):
            p = args[i]
            sub_prefix = "%s/%d" % (prefix, i)
            type_p = type(p)
            if isinstance(p, ScheduledParam):
                self.schedule_param(sub_prefix, p)
            elif type_p == list or type_p == tuple:
                self.schedule_params(sub_prefix, _spc_depth+1, *p)
            elif type_p == dict:
                self.schedule_params(sub_prefix, _spc_depth+1, **p)
            elif type_p == int or type_p == float or isinstance(p, FloatParam):
                self.collect_numeric_param(sub_prefix, p)
        for key in kwargs:
            p = kwargs[key]
            sub_prefix = "%s/%s" % (prefix, key)
            type_p = type(p)
            if isinstance(p, ScheduledParam):
                self.schedule_param(sub_prefix, p)
            elif type_p == list or type_p == tuple:
                self.schedule_params(sub_prefix, _spc_depth+1, *p)
            elif type_p == dict:
                self.schedule_params(sub_prefix, _spc_depth+1, **p)
            elif type_p == int or type_p == float or isinstance(p, FloatParam):
                self.collect_numeric_param(sub_prefix, p)

    def schedule_param(self, prefix, param):
        """
        :param param:
        :type param: ScheduledParam
        :return:
        """
        self._params[prefix] = param

    def collect_numeric_param(self, prefix, param):
        self._numeric_params[prefix] = param

    def set_int_handle(self, int_handle):
        for k in self._params:
            param = self._params[k]
            param.set_int_handle(int_handle)

    def get_params(self):
        return self._params

    def get_numeric_params(self):
        return self._numeric_params


class CappedLinear(ScheduledParam):
    def __init__(self, step, start, end, stepper=None):
        super(CappedLinear, self).__init__(lambda n: end if n > step else start + (end - start) * n / step, stepper,
                                           step=step, start=start, end=end)


class CappedExp(ScheduledParam):
    def __init__(self, step, start, end, stepper=None):
        self._base = np.exp(np.log(1.0*start/end) / step)
        super(CappedExp, self).__init__(lambda n: end if n >= step else end * (self._base ** (step - n)),
                                        stepper, step=step, start=start, end=end)


class Cosine(ScheduledParam):
    def __init__(self, step, start, end, stepper=None):
        super(Cosine, self).__init__(lambda n: start + (1 - math.cos(math.pi * 2 * n / step)) * (end - start)/2,
                                     stepper, step=step, start=start, end=end)

