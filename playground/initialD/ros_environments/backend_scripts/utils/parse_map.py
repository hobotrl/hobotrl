"""Parse map file and extract road segment info.

:author: Gang Xu, Jingchu Liu
:maintainer: Jingchu Liu
"""

import argparse
import traceback
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse OpenDrive xodr file.')
    parser.add_argument('map_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    path_to_map = args.map_dir
    path_to_output = args.output_dir

    tree = ET.parse(path_to_map)
    root = tree.getroot()
    prog = '[parse_map]: '
    roaddata = []
    juncdata = []
    juncdict = {}
    ret = []
    assert root.tag == 'OpenDRIVE'

    # Parse all junctions
    for junc in root.findall('junction'):
        id = junc.get('id')
        linkdata = []
        for connection in junc.findall('connection'):
            connectionID = connection.get('connectingRoad')
            linkdata.append(connectionID)
        juncdict[id] = linkdata
        juncdata += linkdata

    # Parse all roads
    for road in root.findall('road'):
        id = road.get('id')
        roadlen = road.get('length')
        link = road.find('link')
        pre = link.find('predecessor')
        succ = link.find('successor')
        roadx = road.find('planView').find('geometry').get('x')
        roady = road.find('planView').find('geometry').get('y')
        roadhdg = road.find('planView').find('geometry').get('hdg')
        lanes = road.find('lanes')
        laneSec = lanes.find('laneSection')

        left = laneSec.find('left')
        right = laneSec.find('right')
        leftWid = []
        rightWid = []
        if left != None:
            for lane in left.findall('lane'):
                if lane.get('type') == 'driving':
                    try:
                        laneIndex = int(lane.get('id'))
                        lanewidth = float(lane.find('width').get('a'))
                    except Exception as e:
                        traceback.print_exc()
                    leftWid.append(laneIndex*lanewidth-lanewidth/2.0)

        if right != None:
            for lane in right.findall('lane'):
                if lane.get('type') == 'driving':
                    try:
                        laneIndex = int(lane.get('id'))
                        lanewidth = float(lane.find('width').get('a'))
                    except Exception as e:
                        traceback.print_exc()

                    rightWid.append(laneIndex*lanewidth+lanewidth/2.0)

        if pre != None:
            preID = pre.get('elementId')
            preContact = pre.get('contactPoint')
        else:
            print prog+"Road {} has no predecessor.".format(id)
            preID = '000'
            preContact = None

        if succ != None:
            succID = succ.get('elementId')
            succContact = succ.get('contactPoint')
        else:
            print prog+"Road {} has no successor.".format(id)
            succID = '000'
            succContact = None

        if roadlen == None:
            print prog+"Road {} has zero length.".format(id)
            roadlen = 0.0

        roaddata.append([
            preID, id, succID,
            roadlen, leftWid, rightWid, roadx, roady, roadhdg])

    # write outputs
    with open(path_to_output, 'w') as f:
        f.write('route:\n')
        for road in roaddata:
            if road[1] in juncdata:
                # S1->C1->S2 handling
                if road[0] == 'S2':
                    road[0] = 'S1 C1 S2'
                if road[2] == 'S2':
                    road[2] = 'S2 C1 S1'
                # B2->S5 handling
                if road[0] == 'B2':
                    road[0] = 'S5 B2'
                if road[2] == 'B2':
                    road[2] = 'B2 S5'
                # S4->B1 handling
                if road[0] == 'B1':
                    road[0] = 'S4 B1'
                if road[2] == 'B1':
                    road[2] = 'B1 S4'
                # S4->B1 handling
                if road[0] == 'S4':
                    road[0] = 'B1 S4'
                if road[2] == 'S4':
                    road[2] = 'S4 B1'
                ret.append(road)
                f.write('{0} {1} {2}\n'.format(road[0], road[1], road[2]))
        f.write('junc:\n')
        for k, v in juncdict.iteritems():
            f.write('{}:{}\n'.format(k, v))
        f.write('link:\n')
        for road in roaddata:
            f.write('{}, {}, {}\n'.format(road[1], road[0], road[2]))

        f.write('length:\n')
        for road in roaddata:
            f.write('{0} {1}\n'.format(road[1], road[3]))

        f.write('width:\n')
        for road in roaddata:
            f.write('{0} '.format(road[1]))
            for leftWid in road[4]:
                f.write('{0} '.format(leftWid))

            for rightWid in road[5]:
                f.write('{0} '.format(rightWid))

            f.write('\n')

        f.write('geometry:\n')
        for road in roaddata:
            f.write('{}:{} {} {}\n'.format(road[1], road[6], road[7], road[8]))

