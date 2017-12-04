# -*- coding: utf-8 -*-


import tensorflow as tf
from hobotrl.core import BaseAgent
from hobotrl.network import Network
from training import MonitoredTrainingSession


class BaseDeepAgent(BaseAgent):
    def __init__(self, sess=None, graph=None, global_step=None, **kwargs):
        super(BaseDeepAgent, self).__init__(**kwargs)
        self.sess, self.graph = sess, graph
        self.__global_step = global_step
        if self.__global_step is not None:
            with tf.name_scope("update_global_step"):
                # self.__step_input = tf.placeholder(tf.int32, shape=None, name="input_global_step")
                # self.__op_update_step = tf.assign(self.__global_step, self.__step_input)
                self.__op_update_step = tf.assign_add(self.__global_step, 1)
        self.__step_n = 0
        self._network = self.init_network(**kwargs)

    def init_network(self, *args, **kwargs):
        """
        should be overwritten by sub-classes.
        implementation of init_network should return an instance of network.Network,
            to initialize self._network, which could be retrieved by self.network
            Also used by network.DistributedOptimizer to determine local-global variable map.
        :param args:
        :param kwargs:
        :return: Network
        :rtype: Network
        """
        raise NotImplementedError()

    @property
    def network(self):
        return self._network

    def create_session(self, config=None, master="", graph=None, worker_index=0,
                       init_op=None,
                       save_dir=None, restore_var_list=None,
                       save_checkpoint_secs=600,
                       save_summaries_steps=None, save_summaries_secs=None):
        sess = MonitoredTrainingSession(
            master=master, is_chief=(worker_index == 0),
            checkpoint_dir=save_dir,
            restore_var_list=restore_var_list,
            save_checkpoint_secs=save_checkpoint_secs,
            save_summaries_steps=save_summaries_steps,
            save_summaries_secs=save_summaries_secs,
            config=config
        )
        self.set_session(sess)
        return sess

    def get_global_step(self):
        return self.__global_step

    def get_session(self):
        return self.sess

    def set_session(self, sess):
        self.sess = sess
        if self._network is not None:
            self._network.set_session(sess)

    def set_graph(self, graph):
        self.graph = graph

    def get_graph(self):
        return self.graph

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        self.__step_n += 1
        # increment global_step variable by 1
        if self.__global_step is not None:
            # self.sess.run(self.__op_update_step, feed_dict={self.__step_input: self.__step_n})
            self.sess.run(self.__op_update_step)
        # set session for super calls
        if 'sess' not in kwargs:
            kwargs['sess'] = self.sess
        return super(BaseDeepAgent, self).step(state, action, reward, next_state, episode_done, **kwargs)


