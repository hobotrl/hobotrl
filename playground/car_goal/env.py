# -*- coding: utf-8 -*-
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_racing import WINDOW_H, WINDOW_W
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef


class Goal(object):
    """Struct to store local goal."""
    def __init__(self):
        self.pos = (0, 0)
        self.desired_t = 0
        self.issue_t = 0
        self.car = (0, 0, 0)

    @property
    def position_abs(self):
        x, y, yaw = self.car
        trans = np.array(((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw))))
        return np.dot(trans, np.array(self.pos)) + np.array((x, y))

    @property
    def expire_t(self):
        return self.issue_t + self.desired_t

    def __str__(self):
        args = list(self.pos) + [self.desired_t] + [self.issue_t] + list(self.car)
        return "Goal: ({:.3f}, {:.3f}) in {} @ {}. Issue car ({:.3f}, {:.3f}, {:.3f})".format(*args)


class CarRacingGoalWrapper(gym.Wrapper):
    def __init__(self, env, func_control=None, func_reward=None, *args, **kwargs):
        super(CarRacingGoalWrapper, self).__init__(env)
        self.env = env
        self.action_space = spaces.Box(
            low=-np.array((1.0, 1.0)),
            high=np.array((1.0, 1.0))
        )
        # self.observation_space = spaces.Tuple((self.env.observation_space, self.action_space))
        self.observation_space = self.env.observation_space
        self._n_steps = None
        self._goal = Goal()
        self._car = (0, 0, 0)
        if func_control is None:
            self._control_policy = self._default_control_policy
        else:
            self._control_policy = func_control
        if func_reward is None:
            self._reward_function = self._default_reward_function
        else:
            self._reward_function = func_reward

    def _reset(self, **kwargs):
        self._n_steps = 0
        self._goal = Goal()
        self._car = (0, 0, 0)
        state = self.env.reset()
        self.init_marker()
        return state, self._goal

    def _step(self, action):
        self._n_steps += 1
        if action is not None:
            assert self.action_space.contains(action)
            self._goal.pos = (action[0]*20, action[1]*20)
            self._goal.desired_t = 1000  # action[2]
            self._goal.issue_t = self._n_steps
            self._goal.car = (
                self.env.env.car.hull.position.x,
                self.env.env.car.hull.position.y,
                np.mod(3*np.pi/2 + self.env.env.car.hull.angle, 2*np.pi) - np.pi
            )
        self._car = (
            self.env.env.car.hull.position.x,
            self.env.env.car.hull.position.y,
            np.mod(3 * np.pi / 2 + self.env.env.car.hull.angle, 2 * np.pi) - np.pi
        )
        state, reward, done, info = self.env.step(self._control_policy())
        return state, reward, done, info

    def _control_policy(self):
        raise NotImplementedError()

    def _reward_function(self, env_reward):
        raise NotImplementedError

    def _default_control_policy(self):
        if self._goal is not None and \
           self._goal.expire_t >= self._n_steps:
            x, y, yaw = self._car
            gx, gy = self._goal.position_abs
            theta = np.mod(np.arctan2(gy-y, gx-x) - yaw + np.pi, 2 * np.pi) - np.pi

            if_front = np.abs(theta) < np.pi/2
            # if_turn = np.pi/4 < theta < np.pi*3/4 or -np.pi*4/3 < theta < -np.pi/4
            if_turn = True
            if if_front:
                action = (-theta, 1-np.abs(theta), 0) if if_turn else (0, 0.5, 0)
            else:
                action = (0, 0, 0.5)
        else:
            action = (0, 0, 0)
        return action

    def _default_reward_function(self, env_reward):
        return env_reward

    def _render(self, *args, **kwargs):
        position = np.array(self.env.env.car.drawlist[-1].fixtures[0].shape.vertices)
        center = position.mean(axis=0)
        if self._goal is not None:
            position = position - center + np.array((self._goal.pos[0], self._goal.pos[1]))
            self.env.env.car.drawlist[-1].fixtures[0].shape.vertices = position.tolist()
            self.env.env.car.drawlist[-1].fixtures[0].body.transform.position.x = self._goal.car[0]
            self.env.env.car.drawlist[-1].fixtures[0].body.transform.position.y = self._goal.car[1]
            self.env.env.car.drawlist[-1].fixtures[0].body.transform.angle = self._goal.car[2]
        arr = self.env.env._render(*args, **kwargs)
        return arr

    def init_marker(self):
        WHEEL_R = 50
        WHEEL_W = 50
        SIZE = 0.02
        front_k = 1.0
        WHEEL_COLOR = (0.0, 0.0, 0.8)
        WHEEL_POLY = [
            (-WHEEL_W,+WHEEL_R), (+WHEEL_W,+WHEEL_R),
            (+WHEEL_W,-WHEEL_R), (-WHEEL_W,-WHEEL_R)
            ]
        w = self.env.env.car.world.CreateDynamicBody(
            position=(0, 0),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[ (x*front_k*SIZE,y*front_k*SIZE) for x,y in WHEEL_POLY ]),
                density=0.1,
                categoryBits=0x0020,
                maskBits=0x001,
                restitution=0.0)
        )
        w.wheel_rad = front_k * WHEEL_R * SIZE
        w.color = WHEEL_COLOR
        w.gas = 0.0
        w.brake = 0.0
        w.steer = 0.0
        w.phase = 0.0  # wheel angle
        w.omega = 0.0  # angular velocity
        w.skid_start = None
        w.skid_particle = None
        w.tiles = set()
        w.userData = w
        self.env.env.car.drawlist = self.env.env.car.wheels + [self.env.env.car.hull] + [w]

