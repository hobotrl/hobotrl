# -*- coding: utf-8 -*-
"""Generate launch and .obs file with random configuration.

This scrip generates a launch file as well as the accompanying .obs obstacle
configuration file. Ego car and obstacle cars are placed at a random points
on honda map. Obstacle car also takes on random target route, heading direction,
lane offset, starting offset, and target speed.

This scrip relies on a parsed map file with candidate routes, junction
look-up table, predicessor and successor look-up table, as well as route
length, width, and geometry look-up table.

:author: Jingchu Liu
:date: 2017-Sep-4
"""

from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

import numpy as np

import argparse

class CarElement(Element):
    """Car element in obs file element tree.

    :param route_args: (route_ids, )
    :param position_args: (s_offset, lane_offset, target_velocity)
    :param life_args: (appear_time, life_duration, respawn_period)
    """
    def __init__(self, route_args, position_args, life_args, *args, **kwargs):
        super(CarElement, self).__init__('car', *args, **kwargs)
        self.route = None
        self.position = None
        self.life = None
        self.set_route(*route_args)
        self.set_position(*position_args)
        self.set_life(*life_args)

    def set_route(self, route_ids):
        """Set route element.

        :param route_ids: list of ids of route segments.
        """
        route_ids = ','.join(route_ids)
        self.route = Element('route', {'ids': route_ids})
        route_ele = [ele for ele in self.findall('route')]
        for ele in route_ele:
            self.remove(ele)
        self.append(self.route)

    def set_position(self, pos_s, pos_l, pos_v):
        """Set position element.

        :param pos_s: starting s offset. Non-neg float.
        :param pos_l: lane offset. Non-zero float.
        :param pos_v: target velocity. Non-neg float.
        """
        assert pos_v >= 0
        assert pos_l != 0
        assert pos_v >= 0
        pos_attr = {
            "s": "{:.3f}".format(float(pos_s)),
            "l": "{:.3f}".format(float(pos_l)),
            "speed": "{:.3f}".format(float(pos_v))}
        self.position = Element('position', pos_attr)
        position_ele = [ele for ele in self.findall('position')]
        for ele in position_ele:
            self.remove(ele)
        self.append(self.position)

    def set_life(self, t_appear, duration, period):
        """Set life element.

        :param t_appear: time to appear in seconds. Non-neg float.
        :param duration: apearing duration in seconds. Pos float.
        :param period: respawn period. Non-neg float. Zero = off.
        """
        assert t_appear >= 0
        assert duration > 0
        assert period >= 0
        life_attr = {
            "appear_time": "{:.2E}".format(float(t_appear)),
            "duration": "{:.2E}".format(float(duration)),
            "period": "{:.2E}".format(float(period))}
        self.life = Element('life', life_attr)
        life_ele = [ele for ele in self.findall('life')]
        for ele in life_ele:
            self.remove(ele)
        self.append(self.life)

def read_parsed_map(file_name):
    """Read in parsed map and build helper vars."""

    def parse_routes(routes):
        """Parse route lines for list of seg ids."""
        # 'id1, id2, id3\n' -> ['id1', 'id2', 'id3']
        return [route.strip('\n').split() for route in routes]

    def parse_junc(juncs):
        """Parse junction dict."""
        # "junc_id: [link_id1, link_id2]" -> {'junc_id': ['link_id1', 'link_id2']}
        juncs = [junc.split(':') for junc in juncs]
        juncs = [(junc[0], junc[1].translate(None, "[]\' \n").split(','))
                 for junc in juncs]
        return dict(juncs)

    def parse_link(links):
        """Parse links lines for predecessor and successor of links."""
        links = [link.strip('\n').split(',') for link in links]
        preds = [(id, pred.split()) for id, pred, succ in links]
        succs = [(id, succ.split()) for id, pred, succ in links]

        return dict(preds), dict(succs)

    def parse_length(lengths):
        """Parse lengen lines"""
        # 'id: length' -> {id: length_float)} 
        lengths = [length.strip('\n').split() for length in lengths]
        return dict([(id, float(l)) for id, l in lengths])

    def parse_width(widths):
        """Parse width lines."""
        # 'id lane1, lane-1' -> {'id': [float(lane1), float(lane-1)]}
        widths = [width.strip('\n').split() for width in widths]
        return dict([
            (width[0], map(float, width[1:])) for width in widths
        ])

    def parse_geometry(geos):
        """Parse geometry lines."""
        geos = [geo.strip('\n').split(':') for geo in geos]
        geos = [(geo[0], map(float, geo[1].split())) for geo in geos]
        return dict(geos)

    with open(file_name, 'r') as f:
        lines = f.readlines()

    route_begin = lines.index('route:\n')
    junc_begin = lines.index('junc:\n')
    link_begin = lines.index('link:\n')
    length_begin = lines.index('length:\n')
    width_begin = lines.index('width:\n')
    geo_begin = lines.index('geometry:\n')

    list_route = parse_routes(lines[route_begin+1:junc_begin])
    dict_junc = parse_junc(lines[junc_begin+1:link_begin])
    dict_pred, dict_succ = parse_link(lines[link_begin+1:length_begin])
    dict_length = parse_length(lines[length_begin+1:width_begin])
    dict_width = parse_width(lines[width_begin+1:geo_begin])
    dict_geo = parse_geometry(lines[geo_begin+1:])

    return (list_route, dict_junc, dict_pred, dict_succ,
            dict_length, dict_width, dict_geo)

def lane_sign(cur, nxt, dict_junc, dict_pred, dict_succ):
    """Decide the appropriate sign of lane offset values.

    The sign of lane offset values decides on which directions the launch
    starts. However, the sign of lane offsets is defined without consistent
    correspondence with the direction. To resolve this problem, we check
    whether the nxt section on road is the predecessor or successor of the
    current road section, then decide signs accordingly.

    :param cur:
    :param nxt:
    :param dict_junc:
    :param dict_pred:
    :param dict_succ:
    :rtype: float +1.0 or -1.0
    """
    # replace junc ID with appropriate link ID, i.e. Jx -> Lxyz
    preds = [nxt if pred in dict_junc and nxt in dict_junc[pred] else pred
             for pred in dict_pred[cur]]
    succs = [nxt if succ in dict_junc and nxt in dict_junc[succ] else succ
             for succ in dict_succ[cur]]
    # sign logic
    if nxt in preds:
        sign_l = -1
    elif nxt in succs:
        sign_l = 1
    else:
        raise ValueError('Pred or succ error: {}, {}, {}, {}'.format(
            cur, nxt, dict_pred[cur], dict_succ[cur]))
    return sign_l

def gen_single_car(
    list_route, dict_junc, dict_pred, dict_succ, dict_length, dict_width):
    """Randomly generate params for a single car.

    :param list_route: list of candidate route. Each is list of route ids.
    :param dict_junc: map junction id to list of link ids.
    :param dict_pred: map link id to list of predecessor road ids.
    :param dict_succ: map link id to list of successor road ids.
    :param dict_length: map road id to road length (meters).
    :param dict_width: map road id to offset of lanes (meters).
    """
    # randomly select a route
    route_ids = np.random.choice(list_route)
    # decide the sign of lane offset values for current heading direction
    cur, nxt = route_ids[0], route_ids[1]
    sign_l = lane_sign(cur, nxt, dict_junc, dict_pred, dict_succ)
    # uniformly randomly select an offset on the 1st road segment.
    # Note: only select from the first three quaters to avoid starting
    # too close to the insersections, which by large chance not a valid
    # test case. 
    pos_s = np.random.rand()*dict_length[route_ids[0]]*0.75
    # randomly select lane offset from current direction
    pos_l = np.random.choice(
        filter(lambda x: x*sign_l>=0, dict_width[route_ids[0]]))
    # randomly set speed
    pos_v = np.random.rand()*5.0 + 5.0  # Uniform [5.0, 10.0)

    # set life
    life_tappear = 0.0
    life_duration = 4.0e5
    life_period = np.random.rand()*30.0 + 30.0  # Uniform [30.0, 60)

    return ((route_ids,), (pos_s, pos_l, pos_v),
            (life_tappear, life_duration, life_period))

def project_coord(s, x0, y0, hdg, l):
    """Get coordinate of a point along a straight road.

    Used to get the coordinate of a point that is 's' meters away from
    the starting point on a straight road.

    :param s: distance along chord line from the road start.
    :param x0: road start x-coord.
    :param y0: road start y-coord.
    :param hdg: heading angle of the road segment.
    :param l: lane offset of the starting point.
    :rtype: x, y coord. and heading angle of starting point.
    """
    x = x0 + s*np.cos(hdg) - l*np.sin(hdg)
    y = y0 + s*np.sin(hdg) + l*np.cos(hdg)
    carHdg = hdg + np.pi*(l<0)
    if carHdg > np.pi:
        carHdg -= 2*np.pi
    return x, y, carHdg


if __name__=='__main__':
    parser = argparse.ArgumentParser('Generate random planning launch file.')
    parser.add_argument('map_file')
    parser.add_argument('ws_dir')
    parser.add_argument('launch_template_file')
    parser.add_argument('n_obs', type=int)
    args = parser.parse_args()

    obs = Element('obstacle')
    map_file = args.map_file  # './road_segment_info.txt' 
    list_route, dict_junc, dict_pred, dict_succ, \
        dict_length, dict_width, dict_geo = read_parsed_map(map_file)

    ws_dir = args.ws_dir
    if ws_dir[-1] != '/':
        ws_dir += '/'
    planning_path = ws_dir + 'src/Planning/planning/'

    # Obstacles
    num_obs = args.n_obs
    for _ in range(num_obs):
        obs.append(CarElement(*gen_single_car(
            list_route, dict_junc, dict_pred, dict_succ,
            dict_length, dict_width)))
    tree = ET.ElementTree()
    tree._setroot(obs)
    tree.write(planning_path+'config/honda_dynamic_obs.obs',
               encoding='utf-8', xml_declaration=True)

    # Ego car
    (route_ids,), (pos_s, pos_l, _), _ = gen_single_car(
            list_route, dict_junc, dict_pred, dict_succ,
            dict_length, dict_width)
    x0, y0, hdg = dict_geo[route_ids[0]]
    x, y, carHdg = project_coord(pos_s, x0, y0, hdg, pos_l)
    print ("[gen_dynamic_launch.py]: car launch: route {}, lane {},"
           "offset {}, x {}, y {}, heading {}").format(
               route_ids, pos_l, pos_s, x0, y0, carHdg)

    tree = ET.parse(args.launch_template_file)
    root = tree.getroot()
    for param in root.findall('param'):
        if param.get('name')=='/route':
            param.set('value', ','.join(route_ids))
        if param.get('name')=='/obstacles3/filename':
            param.set('value', planning_path+'config/honda_dynamic_obs.obs')
    for include in root.findall('include'):
        for arg in include.findall('arg'):
            if arg.get('name')=='car_pos':
                arg.set('value', '-x {} -y {} -z 0.0 -Y {}'.format(x, y, carHdg))
    tree.write(planning_path+'launch/honda_dynamic_obs.launch',
               encoding='utf-8', xml_declaration=True)


