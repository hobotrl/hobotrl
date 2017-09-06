# -*- coding: utf-8 -*-
"""Server node of a distributed simulation environment.

:file_name: server.py
:author: Jingchu Liu
:data: 2017-09-05
"""

# Basic python
import importlib
import signal
import time
import sys
import os
import traceback
# comms
import zmq
import dill
# Multiprocessing
import multiprocessing
# To find env packages
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class DrivingSimulatorEnvServer(multiprocessing.Process):
    def __init__(self, port):
        self.port = port
        self.context = None
        self.socket = None
        super(DrivingSimulatorEnvServer, self).__init__()

    def run(self):
        if self.socket is not None:
           self.socket.close()
        if self.context is not None:
            self.context.term()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % self.port)

        try:
            while True:
                msg_type, msg_payload = self.socket.recv_pyobj()
                print msg_payload
                if msg_type == 'start':
                    print msg_payload
                    env_kwargs = {}
                    for key, value in msg_payload.iteritems():
                        env_kwargs[key] = dill.loads(value)
                    if 'env_class_name' in kwargs:
                        env_class_name = env_kwargs['env_class_name']
                        del env_kwargs['env_class_name']
                    else:
                        env_class_name = 'core.DrivingSimulatorEnv'
                    package_name = '.'.join(env_class_name.split('.')[:-1])
                    class_name = env_class_name.split('.')[-1]
                    DrivingSimulatorEnv = getattr(
                        importlib.import_module(package_name), class_name)
                    self.env = DrivingSimulatorEnv(**env_kwargs)
                    self.socket.send_pyobj(('start', None))
                elif msg_type == 'reset':
                    msg_rep = self.env.reset()
                    self.socket.send_pyobj(('reset', msg_rep))
                elif msg_type == 'step':
                    msg_rep = self.env.step(*msg_payload)
                    self.socket.send_pyobj(('step', msg_rep))
                elif msg_type == 'exit':
                    self.env.exit()
                    self.env = None
                    self.socket.send_pyobj(('exit', None))
                else:
                    raise ValueError(
                        'EnvServer: unrecognized msg type {}.'.format(msg_type))

        except:
            traceback.print_exc()
        finally:
            self.socket.close()
            self.context.term()

    def exit(self):
        try:
            self.socket.close()
        except:
            pass
        try:
            self.context.term()
        except:
            pass
        try:
            self.env.exit()
        except:
            pass

