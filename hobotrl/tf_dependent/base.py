# -*- coding: utf-8 -*-


import tensorflow as tf
from hobotrl.core import BaseAgent


class BaseDeepAgent(BaseAgent):
    def __init__(self, sess=None, graph=None, global_step=None, **kwargs):
        super(BaseDeepAgent, self).__init__(**kwargs)
        self.sess, self.graph = sess, graph
        self.__sv = None
        self.__global_step = global_step
        self.__global_step = global_step
        if self.__global_step is not None:
            self.sess = None
            with tf.name_scope("update_global_step"):
                self.__step_input = tf.placeholder(tf.int32, shape=None, name="input_global_step")
                self.__op_update_step = tf.assign(self.__global_step, self.__step_input)
        self.__step_n = 0

    def init_supervisor(self, graph=None, worker_index=0, init_op=None, save_dir=None):
        if init_op is None:
            init_op = tf.global_variables_initializer()
        graph = self.graph if graph is None else graph
        self.__sv = tf.train.Supervisor(graph=graph,
                                        # is_chief=False,
                                        is_chief=(worker_index == 0),
                                        global_step=self.__global_step,
                                        init_op=init_op,
                                        logdir=save_dir if worker_index == 0 else None,
                                        save_summaries_secs=0)
        return self.__sv

    def get_supervisor(self):
        return self.__sv

    def get_global_step(self):
        return self.__global_step

    def get_session(self):
        return self.sess

    def set_session(self, sess):
        self.sess = sess

    def set_graph(self, graph):
        self.graph = graph

    def get_graph(self):
        return self.graph

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        self.__step_n += 1
        # increment global_step variable by 1
        if self.__global_step is not None:
            self.sess.run(self.__op_update_step, feed_dict={self.__step_input: self.__step_n})
        # set session for super calls
        if 'sess' not in kwargs:
            kwargs['sess'] = self.sess
        return super(BaseDeepAgent, self).step(state, action, reward, next_state, episode_done, **kwargs)
