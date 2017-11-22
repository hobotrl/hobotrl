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
from gen_launch_dynamic import *

if __name__=='__main__':
    parser = argparse.ArgumentParser('Generate random planning launch file.')
    parser.add_argument('map_file')
    parser.add_argument('ws_dir')
    parser.add_argument('launch_template_file')
    parser.add_argument('n_obs', type=int)
    parser.add_argument(
        '--random_n_obs', dest='random_n_obs', action='store_true'
    )
    parser.set_defaults(random_n_obs=False)
    parser.add_argument(
        '--include_short_segment', dest='include_short', action='store_true'
    )
    parser.set_defaults(include_short=False)
    args = parser.parse_args()

    obs = Element('obstacle')
    map_file = args.map_file  # './road_segment_info.txt' 
    list_route, dict_junc, dict_pred, dict_succ, \
        dict_length, dict_width, dict_geo = read_parsed_map(map_file)

    ws_dir = args.ws_dir
    if ws_dir[-1] != '/':
        ws_dir += '/'
    planning_path = ws_dir + 'src/Planning/planning/'

    # === Ego car ===
    while True:
        (route_ids,), (pos_s, pos_l, _), _ = gen_single_car(
                list_route, dict_junc, dict_pred, dict_succ,
                dict_length, dict_width)
        if args.include_short or route_ids[0] not in ['S7', 'S10', 'S9', 'S11', 'S12']:
            break
    pos_s = adjust_ego_start_pos(
        route_ids, pos_l, dict_length, dict_geo, dict_width)

    # start
    x0, y0, hdg = dict_geo[route_ids[0]]
    x, y, carHdg = project_coord(pos_s, x0, y0, hdg, pos_l)
    # destination
    xd, yd = gen_destination_coord(route_ids, dict_length, dict_geo, dict_width)
    print ("[gen_dynamic_launch.py]: car launch: \n"
           "    route {}, lane {}, offset {}/{}, \n"
           "    x {}, y {}, heading {}, \n"
           "    destination (x {}, y {})").format(
               route_ids, pos_l, pos_s, dict_length[route_ids[0]],
               x0, y0, carHdg, xd, yd)

    tree = ET.parse(args.launch_template_file)
    root = tree.getroot()
    for param in root.findall('param'):
        if param.get('name')=='/route':
            param.set('value', ','.join(route_ids))
        if param.get('name')=='/obstacles3/filename':
            param.set('value', planning_path+'config/honda_dynamic_obs.obs')
        if param.get('name')=='/car/dest_coord_x':
            param.set('value', str(xd))
        if param.get('name')=='/car/dest_coord_y':
            param.set('value', str(yd))
    for include in root.findall('include'):
        for arg in include.findall('arg'):
            if arg.get('name')=='car_pos':
                arg.set('value', '-x {} -y {} -z 0.0 -Y {}'.format(x, y, carHdg))
    tree.write(planning_path+'launch/honda_dynamic_obs.launch',
               encoding='utf-8', xml_declaration=True)

    # Obstacles
    num_obs = args.n_obs
    ego_route = route_ids
    ego_link = [l for l in ego_route if 'L' in l][0]
    ego_junc = [j for j in dict_junc if ego_link in dict_junc[j]][0]
    adj_routes = [
        r for r in list_route
        if [rr for rr in r if 'L' in rr][0] in dict_junc[ego_junc]
        and r != ego_route
    ]

    if args.random_n_obs:
        num_obs = np.random.randint(low=0, high=num_obs+1)
    print "[gen_dynamic_launch.py]: {} obstacle cars.".format(num_obs)
    num_same = num_obs/8
    num_adj = num_obs - num_same
    v_range = (2, 10)
    period_range = (40, 120)
    for _ in range(num_same):
        obs.append(
            CarElement(
                *gen_single_car(
                    [ego_route], dict_junc, dict_pred, dict_succ,
                    dict_length, dict_width,
                    v_range, period_range
                )
            )
        )
    for _ in range(num_adj):
        obs.append(
            CarElement(
                *gen_single_car(
                    adj_routes, dict_junc, dict_pred, dict_succ,
                    dict_length, dict_width,
                    v_range, period_range
                )
            )
        )
    tree = ET.ElementTree()
    tree._setroot(obs)
    tree.write(planning_path+'config/honda_dynamic_obs.obs',
               encoding='utf-8', xml_declaration=True)


