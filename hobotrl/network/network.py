# -*- coding: utf-8 -*-


import logging

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import hobotrl.utils as utils


class Utils(object):

    @staticmethod
    def layer_fcs(input_var, shape, out_count, activation_hidden=tf.nn.elu, activation_out=None, l2=0.0001,
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
               activation=tf.nn.elu, l2=1e-4, var_scope=""):
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
                activation=tf.nn.elu, l2=1e-4, var_scope=""):
        out = input_var
        with tf.variable_scope(var_scope):
            for i in range(len(shape)):
                s = shape[i]
                filter_n, kernel_n, strides_n = s
                out = Utils.conv2d(out, h=kernel_n, w=kernel_n, out_channel=filter_n,
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

    @staticmethod
    def scope_vars(scope, graph_key=tf.GraphKeys.TRAINABLE_VARIABLES):
        """
        Get variables inside a scope
        The scope can be specified as a string

        Parameters
        ----------
        scope: str or VariableScope
            scope in which the variables reside.
        trainable_only: bool
            whether or not to return only the variables that were marked as trainable.

        Returns
        -------
        vars: [tf.Variable]
            list of variables in `scope`.
        """
        return tf.get_collection(
            graph_key,
            scope=scope if isinstance(scope, str) else scope.name
        )

    @staticmethod
    def abs_var_scope(var_scope):
        current = tf.get_variable_scope().name
        return current + "/" + var_scope if current != "" else var_scope

    @staticmethod
    def relative_var_scope(abs_var_scope):
        current = tf.get_variable_scope().name
        if abs_var_scope.find(current) != 0:
            raise IndexError("cannot access scope[%s] from current scope[%s]!" % (abs_var_scope, current))
        if current == abs_var_scope:
            return ""
        return abs_var_scope[len(current)+1:]


class Function(object):
    """
    Function is defined as y = f(x|theta): outputs y given x, possibly with parameter theta.

    """
    def __call__(self, *args, **kwargs):
        """
        outputs y gi
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()


class NetworkSymbol(object):
    """
    represents a Tensor/operation in tf network.
    Holds reference to its Tensor, network, and name.
    """
    def __init__(self, op, name, network):
        """
        :param op:
        :type op: tf.Tensor
        :param name:
        :type name: str
        :param network:
        :type network: Network
        """
        super(NetworkSymbol, self).__init__()
        self._op, self._name, self._network = op, name, network

    @property
    def op(self):
        return self._op

    @property
    def name(self):
        return self._name

    @property
    def network(self):
        return self._network


class Network(object):
    """
    tensorflow network.

    """
    def __init__(self, inputs, network_creator, var_scope, name_scope=None, reuse=False):
        """

        :param inputs:
        :param network_creator:
        :param var_scope: string variable scope. derives absolute variable scope,
            for later variable sharing & retrieving
        :param name_scope:
        :param reuse:
        """
        self._var_scope, self._f_creator, self._inputs = var_scope, network_creator, inputs
        self._abs_var_scope = Utils.abs_var_scope(var_scope)
        with tf.variable_scope(var_scope, reuse=reuse):
            if name_scope is not None:
                with tf.name_scope(name_scope):
                    net = network_creator(inputs)
            else:
                net = network_creator(inputs)
        self._symbols = dict([(k, NetworkSymbol(net[k], k, self)) for k in net])
        self._variables = Utils.scope_vars(self._abs_var_scope)
        self._sess = None
        logging.warning("Network[vs=%s,abs_vs=%s,var=%s,symbols=%s]",
                        self._var_scope, self.abs_var_scope,self.variables, self._symbols)

    def __getitem__(self, item):
        """
        return NetworkSymbol defined in this network, by network_creator
        :param item: name of the symbol to retrieve
        :type item: str
        :return: symbol
        :rtype: NetworkSymbol
        """
        return self._symbols.get(item)

    def __call__(self, inputs, name_scope="", *args, **kwargs):
        """
        :param inputs:
        :param args:
        :param kwargs:
        :return: another network created by this network's network_creator and possibly share weights
        """
        # in order to share weight we need to get access to original abs_var_scope,
        # because current context variable_scope may be different.
        # so we have to derive current relative var_scope, from current var_scope and abs_var_scope
        return Network(inputs, self._f_creator, self.relative_var_scope, name_scope, True)

    @property
    def variables(self):
        # return trainable variables in this network
        return self._variables

    @property
    def var_scope(self):
        return self._var_scope

    @property
    def abs_var_scope(self):
        """
        absolute var scope = tf.current_scope + var_scope.
        :return:
        """
        return self._abs_var_scope

    @property
    def relative_var_scope(self):
        """
        relative var scope = abs_var_scope - tf.current_scope.
        :return:
        """
        return Utils.relative_var_scope(self._abs_var_scope)

    @property
    def inputs(self):
        return self._inputs

    @property
    def session(self):
        return self._sess

    def set_session(self, sess):
        self._sess = sess


class NetworkSyncer(object):
    def __init__(self, src_network, dest_network):
        super(NetworkSyncer, self).__init__()
        self._src_network, self._dest_network = src_network, dest_network
        self._input_rate = tf.placeholder(tf.float32, name="input_sync_rate")
        self._op_sync = [tf.assign_add(target, (learn - target) * self._input_rate)
                         for target, learn in zip(self._dest_network.variables, self._src_network.variables)]
        self._op_sync = tf.group(*self._op_sync, name="sync")

    def sync(self, sess, rate):
        sess.run([self._op_sync], feed_dict={self._input_rate: rate})


class NetworkWithTarget(Network):
    def __init__(self, inputs, network_creator, var_scope, target_var_scope, name_scope=None, reuse=False):
        super(NetworkWithTarget, self).__init__(inputs, network_creator, var_scope, name_scope, reuse)
        self._target = Network(inputs, network_creator, target_var_scope, name_scope, reuse)
        with tf.name_scope("sync_target"):
            self._syncer = NetworkSyncer(self, self._target)

    def sync_target(self, sess, rate):
        return self._syncer.sync(sess, rate)

    def __call__(self, inputs, name_scope="", *args, **kwargs):
        return NetworkWithTarget(inputs, self._f_creator,
                                 self.relative_var_scope, self._target.relative_var_scope,
                                 name_scope, reuse=True)

    @property
    def target(self):
        return self._target

    def set_session(self, sess):
        super(NetworkWithTarget, self).set_session(sess)
        self._target.set_session(sess)


class NetworkFunction(Function):
    def __init__(self, outputs, inputs=None, variables=None):
        """
        Function with a network backend.

        :param outputs: a symbol, or a list of symbols, or a dict of symbols, or a network, representing outputs of this function
                if outputs is a NetworkSymbol, __call__ to this function will return a ndarray;
                if output is a list of NetworkSymbol, __call__ will return a list of ndarray;
                if output is a dict of NetworkSymbol, __call__ will return a dict of ndarray.
                if output is a Network, this function will take network's input as input, and network's all symbol as symbol.
        :type outputs: NetworkSymbol, list[NetworkSymbol], dict(str, NetworkSymbol)

        :param inputs: list of input symbol
                if inputs is a list, this function can be called with *args;
                if inputs is a dict, this function can be called with **kwargs.
                if inputs is None, then inputs is derived from the input of symbol's network
        :type inputs: list, dict
        """
        super(NetworkFunction, self).__init__()

        if isinstance(outputs, Network):
            outputs, inputs = outputs._symbols, outputs._inputs

        self._output_type = type(outputs)
        if self._output_type == NetworkSymbol:
            self._outputs = [outputs]
        elif self._output_type == list:
            self._outputs = outputs
        elif self._output_type == dict:
            self._output_names = outputs.keys()
            self._outputs = [outputs[k] for k in self._output_names]
            self._output_dict = outputs

        self._network = self._outputs[0].network

        if inputs is None:
            inputs = self._network._inputs
        self._inputs = inputs
        if variables is not None:
            self._variables = variables
        else:
            self._variables = self._network.variables
        self._sess = None

    def output(self, index=None):
        """
        :param index:
        :return:
        :rtype: NetworkSymbol
        """
        index = 0 if index is None else index
        if type(index) == int:
            return self._outputs[index]
        elif type(index) == str:
            return self._output_dict[index]
        else:
            raise NotImplementedError("unknown index:%s" % index)

    @property
    def outputs(self):
        if self._output_type == NetworkSymbol or self._output_type == list:
            return self._outputs
        else:
            return self._output_dict

    @property
    def inputs(self):
        return self._inputs

    @property
    def variables(self):
        return self._variables

    @property
    def network(self):
        return self._network

    @property
    def session(self):

        sess = self._sess if self._sess is not None else self._network.session
        return sess

    def __call__(self, *args, **kwargs):
        results = self.session.run([symbol.op for symbol in self._outputs], feed_dict=self.input_dict(*args, **kwargs))
        if self._output_type == NetworkSymbol:
            return results[0]
        elif self._output_type == list:
            return results
        elif self._output_type == dict:
            return dict(zip(self._output_names, results))

    def input_dict(self, *args, **kwargs):
        """
        constructs feed_dict for sess.run()ing underlying networks.
        :param args:
        :param kwargs:
        :return:
        """
        feed_dict = {}
        if type(self._inputs) == dict:
            if len(self._inputs) == 1 and len(kwargs) == 0:
                    feed_dict[self._inputs.values()[0]] = args[0]
            else:
                for k in self._inputs:
                    holder = self._inputs[k]
                    value = kwargs[k]
                    feed_dict[holder] = value
        else:
            feed_dict = dict([(holder, value) for holder, value in zip(self._inputs, args)])
        return feed_dict

    def set_session(self, sess):
        self._sess = sess


class UpdateOperation(object):
    """
    represents network update operation.
    """
    def __init__(self, op_list, var_list=None):
        """
        :param op_list: tf.Tensor, or list of tf.Tensors
        :param feed_dict: feed_dict required for tf.Session.run
        :param var_list: optional list of parameters on which update is performed
        :param fetch_dict: optional dict of values or tf.Tensor's which should return to invoker by this update
        """
        super(UpdateOperation, self).__init__()
        self._op_list, self._var_list, = op_list, self.merge_list_(var_list)
        self._updater = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def merge_list_(self, var_list):
        var_set = {}
        result = []
        for var in var_list:
            if var in var_set:
                continue
            result.append(var)
            var_set[var] = var
        return result


class MinimizeLoss(UpdateOperation):
    def __init__(self, op_loss, var_list=None):
        super(MinimizeLoss, self).__init__(op_loss, var_list)


class ApplyGradient(UpdateOperation):
    def __init__(self, op_list):
        super(ApplyGradient, self).__init__(op_list)


class UpdateRun(object):
    def __init__(self, feed_dict=None, fetch_dict=None):

        feed_dict = {} if feed_dict is None else feed_dict
        fetch_dict = {} if fetch_dict is None else fetch_dict

        self._feed_dict, self._fetch_dict = feed_dict, fetch_dict
        self._updater = None


class NetworkUpdater(object):
    """
    Updater defines update method for a parameterized Function or Network.
    the ctor should construct symbols for calculating updates.
    the `update` method should return an instance of NetworkUpdate
    """

    def __init__(self):
        super(NetworkUpdater, self).__init__()
        self.name = None

    def declare_update(self):
        """
        :return: an instance of UpdateOperation representing the actual update operation of this NetworkUpdater.
        :rtype: UpdateOperation
        """
        raise NotImplementedError()

    def update(self, sess, *args, **kwargs):
        """

        :param sess:
        :param args:
        :param kwargs:
        :return: an instance of UpdateRun representing the run-time parameter of update operation.
        :rtype: UpdateRun
        """
        raise NotImplementedError()


class NetworkOptimizer(object):
    """
    Perform optimize steps.
    An optimize step consists of one or more Updater.update().
    Typical usage:

    construction phase:
        network_optimizer.add_updater(OneStepTD(self.learn_q, self.target_q), name="td")
        network_optimizer.add_updater(L2(self.learnable_network), name="l2")
        network_optimizer.compile()

    running phase:
        network_optimizer.update("td", self.sess, batch)
        network_optimizer.update("l2", self.sess)
        network_optimizer.optimize_step()

    """
    def add_updater(self, updater, weight=1.0, name=None):
        """
        Appends NetworkUpdater to this optimizer
        :param updater:
        :type updater: NetworkUpdater
        :param weight: optional weight of updater
        :param name: optional name of updater. Named updaters can be retrieved by updater(name)
        :return: None
        """
        raise NotImplementedError()

    def update(self, updater_name=None, *args, **kwargs):
        """
        call named updater. Actually a UpdaterWrapper is returned.
        :param updater_name:
        :return: NetworkUpdater
        """
        raise NotImplementedError()

    def compile(self):
        """
        Merge all update operations from all updaters into one optimize operation.
        No updaters can be added by add_updater after compile()
        :return:
        """
        raise NotImplementedError()

    def optimize_step(self, sess):
        """
        Performs one step of optimize operation generated by compile().
        :param sess:
        :return:
        """
        raise NotImplementedError()


class BaseNetworkOptimizer(NetworkOptimizer):
    def __init__(self, grad_clip=None, name=""):
        super(BaseNetworkOptimizer, self).__init__()
        self._name = name
        self._list_updater, self._dict_updater, self._updater_weights, self._updater_labels = [], {}, {}, {}
        self._name_scope = tf.name_scope("NetworkOptimizer%s" % self._name)
        self._optimize_op = None
        self._update_runs = []
        self._grad_clip = grad_clip
        self._default_optimizer = tf.train.AdamOptimizer()

    def add_updater(self, updater, weight=1.0, name=None):
        if self._optimize_op is not None:
            raise RuntimeError("no updater can be added after compile()!")
        if name is None:
            self._list_updater.append(updater)
            self._updater_labels[updater] = self.gen_updater_label_(updater, str(len(self._list_updater)-1))
        else:
            self._dict_updater[name] = updater
            self._updater_labels[updater] = self.gen_updater_label_(updater, name)

        self._updater_weights[updater] = weight

    def update(self, updater_name=None, *args, **kwargs):
        updater = None
        if updater_name is None:
            assert(len(self._list_updater) + len(self._dict_updater) == 1)
            if len(self._list_updater) > 0:
                updater = self._list_updater[0]
            elif len(self._dict_updater) > 0:
                for k in self._dict_updater:
                    updater = self._dict_updater[k]
                    break
        else:
            updater = self._dict_updater[updater_name]
        update_run = updater.update(*args, **kwargs)
        update_run._updater = updater
        self.collect_update_run(update_run)

    def gen_updater_label_(self, updater, label):
        return updater.__class__.__name__ + "/" + label

    def collect_update_run(self, update_run):
        self._update_runs.append(update_run)

    def compile(self):
        if self._optimize_op is not None:
            raise RuntimeError("compile() can be invoked only once!")
        updates = []
        for updater in self._list_updater:
            update = updater.declare_update()
            update._updater = updater
            updates.append(update)
        for k in self._dict_updater:
            updater = self._dict_updater[k]
            update = updater.declare_update()
            update._updater = updater
            updates.append(update)
        self._optimize_op = self.create_optimize_op_(updates)

    def optimize_step(self, sess):
        result = self.run_(sess, self._optimize_op, self._update_runs)
        self._update_runs = []
        return result

    def create_optimize_op_(self, updates):
        with tf.name_scope("optimizers"):
            grads_and_vars = self.compute_gradients_op_(updates)
            return self.apply_gradients_op_(grads_and_vars)

    def compute_gradients_op_(self, updates):
        """

        :param updates:
        :type updates: UpdateOperation
        :return:
        """
        up_losses = filter(lambda up: type(up) == MinimizeLoss, updates)
        up_grads = filter(lambda up: type(up) == ApplyGradient, updates)
        weighted_grads = []

        # merge losses with same var_list
        varlist_losses = {}
        for loss_update in up_losses:
            key_list = utils.hashable_list(loss_update._var_list)
            if key_list in varlist_losses:
                losses = varlist_losses[key_list]
            else:
                losses = []
                varlist_losses[key_list] = losses
            losses.append(loss_update)

        for varlist in varlist_losses:
            losses = varlist_losses[varlist]
            loss = tf.add_n([self._updater_weights[up_loss._updater] * up_loss._op_list for up_loss in losses])
            grads_vars = self._default_optimizer.compute_gradients(loss, var_list=varlist)
            weighted_grads.extend(grads_vars)

        for grad_update in up_grads:
            grads_vars = grad_update._op_list
            for i, (grad, var) in enumerate(grads_vars):
                if grad is not None:
                    grads_vars[i] = (grad * self._updater_weights[grad_update._updater], var)
            weighted_grads.extend(grads_vars)
        # merge weighted_grads
        var_indices = {}
        merged_grad_vars = []
        for grad, var in weighted_grads:
            if var in var_indices:
                var_index = var_indices[var]
                old_grad, var = merged_grad_vars[var_index]
                if grad is None:
                    grad = old_grad
                elif old_grad is None:
                    pass
                else:  # both not None
                    grad = grad + old_grad
                merged_grad_vars[var_index] = (grad, var)
            else:
                var_index = len(var_indices)
                var_indices[var] = var_index
                merged_grad_vars.append((grad, var))

        # Clip gradients 
        # TODO: grad norm is also an useful piece of info. Should be able to
        #       fetch it.
        if self._grad_clip is not None:
            clipped_grads, _ = tf.clip_by_global_norm(
                [g for g, v in merged_grad_vars], self._grad_clip)
            clipped_grad_vars = [(clipped_grads[i], v) for i, (g, v) in
                                 enumerate(merged_grad_vars) if g is not None]
            return clipped_grad_vars
        else:
            return merged_grad_vars

    def run_(self, sess, op, update_runs):
        """
        actually run op with session, together with all fetch_dict defined in updates.
        :param sess:
        :param op: tf.Tensor, or list of tf.Tensor. the op to run.
        :param update_runs:
        :type update_runs: list[UpdateRun]
        :return:
        """
        if type(op) == list:
            op_list = op
        else:
            op_list = [op]
        feed_dict = {}
        fetch_ops, fetch_names = [], []
        prefetched = {}
        for update_run in update_runs:
            # merge parameters
            feed_dict.update(update_run._feed_dict)  # todo value check
            if update_run._fetch_dict is not None:
                for k in update_run._fetch_dict:
                    fetch_name = self._updater_labels[update_run._updater] + "/" + k
                    fetch_var = update_run._fetch_dict[k]
                    if type(fetch_var) == np.ndarray or isinstance(fetch_var, float) or isinstance(fetch_var, int):
                        prefetched[fetch_name] = fetch_var
                        continue
                    fetch_ops.append(fetch_var)
                    fetch_names.append(fetch_name)
        results = sess.run(fetch_ops + op_list, feed_dict=feed_dict)
        if len(fetch_ops) > 0:
            fetch_results = results[:len(fetch_ops)]
            result_dict = dict(zip(fetch_names, fetch_results))
        else:
            result_dict = {}
        result_dict.update(prefetched)
        return result_dict

    def apply_gradients_op_(self, grads_and_vars):
        raise NotImplementedError()


class LocalOptimizer(BaseNetworkOptimizer):
    """
    apply gradients to local variables
    """
    def __init__(self, optimizer=None, grad_clip=None, name=""):
        """

        :param optimizer:
        :type optimizer: tf.train.Optimizer
        :param grad_clip:
        :param name:
        """
        super(LocalOptimizer, self).__init__(grad_clip, name)
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(1e-4)
        self._optimizer = optimizer

    def apply_gradients_op_(self, grads_and_vars):
        return self._optimizer.apply_gradients(grads_and_vars)


class DistributedOptimizer(BaseNetworkOptimizer):
    """
    apply gradients to global variables, and pull newly updated variables from global
    """
    def __init__(self, grad_clip=None, global_optimizer=None, local_global_var_map=None, name=""):
        """

        :param grad_clip:
        :param global_optimizer:
        :type global_optimizer: tf.train.Optimizer
        :param local_global_var_map:
        :type dict
        :param name:
        """
        super(DistributedOptimizer, self).__init__(grad_clip, name)
        self._global_optimizer = global_optimizer
        self._var_map = local_global_var_map
        logging.warning("var map:%s", local_global_var_map)

    def apply_gradients_op_(self, grads_and_vars):
        with tf.name_scope(self._name):
            grads_and_globalvars = [(grad, self._var_map[var]) for grad, var in grads_and_vars]
            logging.warning("grads_and_global_vars:%s", grads_and_globalvars)
            apply_op = self._global_optimizer.apply_gradients(grads_and_globalvars)
            with tf.control_dependencies([apply_op]):
                pulls = [tf.assign(local_var, global_var) for local_var, global_var in self._var_map.items()]
                pulls = tf.group(*pulls)
                return pulls


class OptimizerPlaceHolder(NetworkOptimizer):
    """
    Placeholder object for actual optimizer.

    Before actual optimizer is set, add_updater() and compile() invocations are cached
    and later applied when actual optimizer is ready.

    updater() and optimize_step() are delegated directly to the actual optimizer.


    OptimizerPlaceHolder is designed to break dependency loop when creating DistributedOptimizer:
    DistributedOptimizer needs local_global_var_map,
    which depends on local network and global network,
    which depends on creation of local agent object and global agent object,
    which depends on NetworkOptimizers in their constructor.

    So we first construct local agent/global agent using OptimizerPlaceHolders,
    retrieve local_global_var_map from local / global networks, construct DistributedOptimizer,
    and set DistributedOptimizer object into placeholder.
    """
    def __init__(self, optimizer=None):
        """
        :param optimizer:
        :type optimizer: NetworkOptimizer
        """
        super(OptimizerPlaceHolder, self).__init__()
        self._optimizer = optimizer
        self._updaters = []
        self._compiled = False

    def update(self, updater_name=None, *args, **kwargs):
        return self._optimizer.update(updater_name, *args, **kwargs)

    def optimize_step(self, sess):
        return self._optimizer.optimize_step(sess)

    def add_updater(self, updater, weight=1.0, name=None):
        if self._compiled:
            raise RuntimeError("no updater can be added after compile()!")
        if self._optimizer is not None:
            self._optimizer.add_updater(updater, weight, name)
        else:
            self._updaters.append((updater, weight, name))

    def compile(self):
        if self._compiled:
            raise RuntimeError("compile() can be invoked only once!")
        if self._optimizer is not None:
            self._optimizer.compile()
        else:
            self._compiled = True

    def set_optimizer(self, optimizer):
        """
        :param optimizer:
        :type optimizer: NetworkOptimizer
        :return:
        """
        if self._optimizer is not None:
            raise RuntimeError("This placeholder already occupied by %s" % str(self._optimizer))
        self._optimizer = optimizer
        if len(self._updaters) > 0:
            for updater, weight, name in self._updaters:
                self._optimizer.add_updater(updater, weight, name)
        if self._compiled:
            self._optimizer.compile()
