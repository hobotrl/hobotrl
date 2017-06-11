# -*- coding: utf-8 -*-

from hobotrl.core import BaseAgent


class BaseDeepAgent(BaseAgent):
    def __init__(self, sess=None, **kwargs):
        super(BaseDeepAgent, self).__init__(**kwargs)
        self.sess = sess

    def get_session(self):
        return self.sess

    def set_session(self, sess):
        self.sess = sess

