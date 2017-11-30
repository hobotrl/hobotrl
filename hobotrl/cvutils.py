# -*- coding: utf-8 -*-

import logging

import numpy as np
import cv2


def to_rad(degree):
    return degree * np.pi / 180


def to_degree(rad):
    return 180 * rad / np.pi


def relative_angle(pos_from, pos_to):
    distance = np.sqrt(np.square(pos_to[:, :, 0] - pos_from[:, :, 0])
                       + np.square(pos_to[:, :, 1] - pos_from[:, :, 1])) + 1e-8
    a1 = np.arccos((pos_to[:, :, 0] - pos_from[:, :, 0]) / distance)
    direction = np.sign(pos_to[:, :, 1] - pos_from[:, :, 1]) * a1
    direction = to_degree(direction)
    return direction


def flow_to_color(flow, max_len=None):
    """
    flow shape: HW2
    :param flow:
    :param max_len:
    :return:
    """
    length = np.sqrt(np.sum(np.square(flow), axis=-1))
    hue = relative_angle(np.zeros_like(flow), flow)
    hue = (hue + 360.0) % 360.0
    value = np.ones_like(hue, dtype=np.float32) * 0.9
    data_max = np.max(length)
    if max_len is None or max_len < data_max:
        max_len = data_max
    saturation = length / max_len
    hsv = np.stack((hue / 360.0 * 360.0, saturation * 1.0, value * 1.0), axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
