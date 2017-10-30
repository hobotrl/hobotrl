"""Parse file and extract info from OpenDrive Map used in Honda Project.

Reads in the OpenDrive map file used in Honda Project and write parsing result
to files.

:author: Gang Xu, Jingchu Liu
:date: 2017-10-16
"""

import argparse
import traceback
import cPickle
import xml.etree.ElementTree as ET

def parse_junction(root):
    """Parse junctions.

    Parse junction elements to find the correspondence between junction id
    and link road id. Return a dict that maps junction id to link road ids
    within it. Also returns a list of all link roads.

    :returns dict_junc_links:
    :returns list_links:
    """
    list_links = []  # link roads within junction
    dict_junc_links = {}  # Junction id to list of link roads

    # Parse junctions and fill dict_junc_links and list_links
    for junc in root.findall('junction'):
        id_junc = junc.get('id')
        local_links = []
        for connection in junc.findall('connection'):
            connectionID = connection.get('connectingRoad')
            local_links.append(connectionID)
        dict_junc_links[id_junc] = local_links
        list_links += local_links

    return dict_junc_links, list_links

def parse_road_links(root):
    """Parse linkage info of roads"""
    str_header = '[parse_map.parse_road_link()]: '
    dict_road_links = {}  # junc_id -> pred_id, succ_id

    for road in root.findall('road'):
        road_id = road.get('id')
        link = road.find('link')
        pre = link.find('predecessor')
        succ = link.find('successor')

        if pre is not None:
            pre_id = pre.get('elementId')
        else:
            pre_id = '000'
            # print str_header+"Road {} has no predecessor.".format(road_id)

        if succ is not None:
            succ_id = succ.get('elementId')
        else:
            succ_id = '000'
            # print str_header+"Road {} has no successor.".format(road_id)

        dict_road_links[road_id] = (pre_id, succ_id)

    return dict_road_links

def parse_road_lanes(root):
    """Parse lane info of roads."""
    dict_road_lanes = {}

    for road in root.findall('road'):
        road_id = road.get('id')
        lanes = road.find('lanes')
        lane_sec = lanes.find('laneSection')

        # left lanes
        left = lane_sec.find('left')
        left_width = []
        if left is not None:
            for lane in left.findall('lane'):
                if lane.get('type') == 'driving':
                    try:
                        lane_id = int(lane.get('id'))
                        lane_width = float(lane.find('width').get('a'))
                    except:
                        traceback.print_exc()
                    left_width.append(lane_id*lane_width-lane_width/2.0)

        # right lanes
        right = lane_sec.find('right')
        right_width = []
        if right is not None:
            for lane in right.findall('lane'):
                if lane.get('type') == 'driving':
                    try:
                        lane_id = int(lane.get('id'))
                        lane_width = float(lane.find('width').get('a'))
                    except:
                        traceback.print_exc()
                    right_width.append(lane_id*lane_width+lane_width/2.0)

        dict_road_lanes[road_id] = (left_width, right_width)

    return dict_road_lanes

def parse_road_geo(root):
    """Parse geometry info of roads.

     junc_id -> roadlen, leftWid, rightWid, roadx, roady, roadhdg
    """
    str_header = '[parse_map.parse_road_geo()]: '
    dict_road_geo = {}
    for road in root.findall('road'):
        road_id = road.get('id')

        road_len = road.get('length')
        road_x = road.find('planView').find('geometry').get('x')
        road_y = road.find('planView').find('geometry').get('y')
        road_hdg = road.find('planView').find('geometry').get('hdg')
        if road_len is None:
            print str_header+"Road {} has zero length.".format(road_id)
            road_len = 0.0

        dict_road_geo[road_id] = (road_len, road_x, road_y, road_hdg)

    return dict_road_geo

def parse_road(root):
    dict_road_links = parse_road_links(root)
    dict_road_lanes = parse_road_lanes(root)
    dict_road_geo = parse_road_geo(root)
    return dict_road_links, dict_road_lanes, dict_road_geo

def parse_road_stoplines_signals(root):
    dict_road_juncdir_stopline = {}
    dict_road_juncdir_signal = {}
    dict_stopline_signal = {}

    for road in root.findall('road'):
        road_id = road.get('id')
        dict_juncdir_stopline = {}  # for the current road
        dict_juncdir_signal = {}
        markers = road.find('markers')
        if markers is None:
            # print ('[parse_map.parse_road_stoplines_signals()]: '
            #        'road {} has no marker(s)').format(road_id)
            continue
        stop_lines = [marker for marker in markers.findall('marker')
                      if marker.get('type') == 'stopLine']
        stop_lines.sort()

        # global mapping from stop_line_id to signal
        for stop_line in stop_lines:
            sl_id = stop_line.get('id')
            sig_ref = stop_line.find('reference').get('signalId')
            dict_stopline_signal[sl_id] = sig_ref

        # local mapping from (junction, direction) to stop_line id
        for stop_line in stop_lines:
            sl_id = stop_line.get('id')
            sl_junc = int(sl_id[2:])/100  # junction id
            sl_dir = (int(sl_id[2:])%100 - 1)%3  # direction
            dict_juncdir_stopline[(sl_junc, sl_dir)] = sl_id
            dict_juncdir_signal[(sl_junc, sl_dir)] = dict_stopline_signal[sl_id]
            # in case signal is valid for both going straight and right turn
            if sl_dir == 2 and (sl_junc, 1) not in dict_juncdir_stopline:
                dict_juncdir_stopline[(sl_junc, 1)] = sl_id
                dict_juncdir_signal[(sl_junc, 1)] = \
                    dict_stopline_signal[sl_id]

        dict_road_juncdir_stopline[road_id] = dict_juncdir_stopline
        dict_road_juncdir_signal[road_id] = dict_juncdir_signal

    return dict_road_juncdir_stopline, \
           dict_road_juncdir_signal, \
           dict_stopline_signal

def parse_link_signals(root,
                       dict_road_juncdir_stopline,
                       dict_road_juncdir_signal,
                       dict_stopline_signal):
    dict_link_signal = {}
    dict_signal_links = {}
    links = [road for road in root.findall('road') if road.get('id')[0] == 'L']

    for link in links:
        link_id = link.get('id')
        pred_id = link.find('link').find('predecessor').get('elementId')
        junc_id = int(link_id[1:])/100
        link_dir = (int(link_id[1:]) % 100 - 1) % 3
        sig_id = dict_road_juncdir_signal[pred_id][(junc_id, link_dir)]
        stopline_id = dict_road_juncdir_stopline[pred_id][(junc_id, link_dir)]
        dict_link_signal[link_id] = sig_id
        if sig_id not in dict_signal_links:
            dict_signal_links[sig_id] = {link_id: True}
        else:
            dict_signal_links[sig_id][link_id] = True

    return dict_link_signal, dict_signal_links

def write_parsing_results(path, road_info, junc_info, signal_info):

    dict_junc_links, list_links = junc_info
    dict_road_links, dict_road_lanes, dict_road_geo = road_info

    with open(path, 'w') as f:
        # available routes
        f.write('route:\n')
        road_ids = dict_road_links.keys()
        road_ids.sort()
        for road_id in road_ids:
            (pre_id, succ_id) = dict_road_links[road_id]
            # Doing this to make sure the route passes by a junction
            if road_id in list_links:
                # S1->C1->S2 handling
                if pre_id == 'S2':
                    pre_id = 'S1 C1 S2'
                if succ_id == 'S2':
                    succ_id = 'S2 C1 S1'
                # B2->S5 handling
                if pre_id == 'B2':
                    pre_id = 'S5 B2'
                if succ_id == 'B2':
                    succ_id = 'B2 S5'
                # S4->B1 handling
                if pre_id == 'B1':
                    pre_id = 'S4 B1'
                if succ_id == 'B1':
                    succ_id = 'B1 S4'
                # S4->B1 handling
                if pre_id == 'S4':
                    pre_id = 'B1 S4'
                if succ_id == 'S4':
                    succ_id = 'S4 B1'
                f.write('{0} {1} {2}\n'.format(pre_id, road_id, succ_id))

        # junction and its connecting roads
        f.write('junc:\n')
        junc_ids = dict_junc_links.keys()
        junc_ids.sort()
        for junc_id in junc_ids:
            links = dict_junc_links[junc_id]
            f.write('{}:{}\n'.format(junc_id, links))

        # links for each road
        f.write('link:\n')
        road_ids = dict_road_links.keys()
        road_ids.sort()
        for road_id in road_ids:
            (pre_id, succ_id) = dict_road_links[road_id]
            f.write('{}, {}, {}\n'.format(road_id, pre_id, succ_id))

        # road length info
        f.write('length:\n')
        road_ids = dict_road_geo.keys()
        road_ids.sort()
        for road_id in road_ids:
            (road_len, _, _, _) = dict_road_geo[road_id]
            f.write('{0} {1}\n'.format(road_id, road_len))

        f.write('width:\n')
        road_ids = dict_road_lanes.keys()
        road_ids.sort()
        for road_id in road_ids:
            (left_width, right_width) = dict_road_lanes[road_id]
            f.write('{0} '.format(road_id))
            for width in left_width:
                f.write('{0} '.format(width))
            for width in right_width:
                f.write('{0} '.format(width))
            f.write('\n')

        f.write('geometry:\n')
        road_ids = dict_road_geo.keys()
        road_ids.sort()
        for road_id  in road_ids:
            (_, road_x, road_y, road_hdg) = dict_road_geo[road_id]
            f.write('{}:{} {} {}\n'.format(road_id, road_x, road_y, road_hdg))

    with open(path+'.signal', 'w') as f:
        cPickle.dump(signal_info, f, )

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser('Parse OpenDrive xodr file.')
    parser.add_argument('map_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    path_to_map = args.map_dir
    path_to_output = args.output_dir

    # parse OpenDrive map and get root
    tree = ET.parse(path_to_map)
    root = tree.getroot()
    assert root.tag == 'OpenDRIVE'

    # Parse junctions
    junc_info = parse_junction(root)

    # Parse roads
    road_info = parse_road(root)

    # Stoplines, signals, and link road correspondence
    dict_road_juncdir_stopline, dict_road_juncdir_signal, dict_stopline_signal = \
        parse_road_stoplines_signals(root)
    dict_link_signal, dict_signal_links = parse_link_signals(
        root, dict_road_juncdir_stopline, dict_road_juncdir_signal, dict_stopline_signal)
    signal_info = (dict_road_juncdir_stopline,
                   dict_road_juncdir_signal,
                   dict_stopline_signal,
                   dict_link_signal,
                   dict_signal_links)

    # write outputs
    write_parsing_results(path_to_output, road_info, junc_info, signal_info)

