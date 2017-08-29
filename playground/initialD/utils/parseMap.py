"""Parse map file and extract road segment info.

Author: Gang Xu
"""

import traceback
import xml.etree.ElementTree as ET

# path_to_map = './honda.xodr'
path_to_map = '/home/lewis/catkin_ws_pirate03_lowres350/src/Map/src/map_api/data/honda_wider.xodr'
tree = ET.parse(path_to_map)
root = tree.getroot()

roaddata = []
juncdata = []
ret = []
if root.tag == 'OpenDRIVE':
    for road in root.findall('road'):
        id = road.get('id')
        roadlen = road.get('length')
        link = road.find('link')
        pre = link.find('predecessor')
        succ = link.find('successor')
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
        else:
            preID = '000'

        if succ != None:
            succID = succ.get('elementId')
        else:
            succID = '000'

        if roadlen == None:
            roadlen = 0.0

        roaddata.append([preID, id, succID, roadlen, leftWid, rightWid])

    for junc in root.findall('junction'):
        for connection in junc.findall('connection'):
            connectionID = connection.get('connectingRoad')
            juncdata.append(connectionID)

else:
    print(root.tag)
    print(type(root.tag))
    print(type('OpenDRIVE'))

path_to_output = './road_segment_info.txt'
with open(path_to_output, 'w') as the_file:
    the_file.write('route:\n')
    for road in roaddata:
        if road[1] in juncdata:
            if road[0] == 'S2':
                road[0] = 'S1 C1 S2'

            if road[2] == 'S2':
                road[2] = 'S2 C1 S1'

            ret.append(road)
            the_file.write('{0} {1} {2}\n'.format(road[0], road[1], road[2]))

    the_file.write('\n')
    the_file.write('length:\n')
    for road in roaddata:
        the_file.write('{0} {1}\n'.format(road[1], road[3]))

    the_file.write('\n')    
    the_file.write('width:\n')
    for road in roaddata:
        the_file.write('{0} '.format(road[1]))
        for leftWid in road[4]:
            the_file.write('{0} '.format(leftWid))

        for rightWid in road[5]:        
            the_file.write('{0} '.format(rightWid))

        the_file.write('\n')


