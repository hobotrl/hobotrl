# -*- coding: utf-8 -*-
"""Server node of a distributed simulation environment.

:file_name: server.py
:author: Jingchu Liu
:data: 2017-09-05
"""

# Basic python
import importlib
import sys
import os
import traceback
import logging
import numpy as np

# comms
import zmq
import dill

# Multiprocessing
import multiprocessing

# To find env packages
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class DrivingSimulatorEnvServer(multiprocessing.Process):
    def __init__(self, port, *args, **args):
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
                        logging.warning(
                            "[DrivingSimulatorEnvServer]: "
                            "deserializing key {}.".format(key)
                        )
                        env_kwargs[key] = dill.loads(value)
                    if 'env_class_name' in env_kwargs:
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


class DrSimDecisionK8SServer(object):
    _version = '20171127'
    _ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
    @staticmethod
    def func_compile_reward(rewards):
        """Server-side reward compilation function.
        Rewards are nested lists. Outter index is the time step, inner index
        is the reward type.

        :param rewards: raw rewards vectors.
        :return:
        """
        rewards = np.mean(np.array(rewards), axis=0)
        return rewards

    @staticmethod
    def func_compile_obs(obss):
        """Server side observation compilation function.

        The observations are:
        1. ('/training/image/compressed', sensor_msgs.msg.CompressedImage'),
        2. ('/decision_result', 'std_msgs.msg.Int16'),
        3. ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),

        1. The last two image frames are max-poolled to combat flickering.
           Image observations are casted as `uint8` to save memory.
        2. Decision result is printed and discarded.
        3. Ego speed is printed and discarded.

        :param obss: original observations.
        :return: compiled obseravation tensor.
        """
        img1, img2 = obss[-1][0], obss[-2][0]
        decision, speed = obss[-1][1], obss[-1][2]
        obs = np.maximum(img1, img2)
        print decision
        print speed
        return obs

    @staticmethod
    def func_compile_action(action):
        return DrSimDecisionK8SServer._ALL_ACTIONS[action]


class DrSimRuleDecisionK8SServer(object):
    _version = '20180103'
    _ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
    @staticmethod
    def func_compile_reward(rewards):
        """Server-side reward compilation function.
        Rewards are nested lists. Outter index is the time step, inner index
        is the reward type.

        :param rewards: raw rewards vectors.
        :return:
        """
        rewards = np.mean(np.array(rewards), axis=0)
        return rewards

    @staticmethod
    def func_compile_obs(obss):
        """Server side observation compilation function.

        The observations are:
        1. ('/training/image/compressed', sensor_msgs.msg.CompressedImage'),
        2. ('/decision_result', 'std_msgs.msg.Int16'),
        3. ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),

        1. The last two image frames are max-poolled to combat flickering.
           Image observations are casted as `uint8` to save memory.
        2. Decision result is printed and discarded.
        3. Ego speed is printed and discarded.

        :param obss: original observations.
        :return: compiled obseravation tensor.
        """
        img1, img2 = obss[-1][0], obss[-2][0]
        decision, speed = obss[-1][1], obss[-1][2]
        obs = np.maximum(img1, img2)
        print decision
        print speed
        return obs, decision

    @staticmethod
    def func_compile_action(action):
        return DrSimDecisionK8SServer._ALL_ACTIONS[action]

