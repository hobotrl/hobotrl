# -*- coding: utf-8 -*-
"""Changes lane marker size for OpenDrive map files.

Author: Jingchu Liu
"""

# Input file
path_in = "./honda.xodr"
f_in = open(path_in, 'r')
lines = f_in.readlines()
f_in.close()

# Output file
path_out = "./honda_wider.xodr"
f_out = open(path_out, 'w')
for line in lines:
    if 'roadMark' in line:
        print line
        line = line.replace('width="1.5e-01"', 'width="3.0e-01"')  # width from -> to
    f_out.write(line)
f_out.close()
