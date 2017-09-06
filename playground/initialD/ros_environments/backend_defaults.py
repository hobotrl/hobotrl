# -*- coding: utf-8 -*-
"""Wrappers for different backends.
Mainly fills in backend commands if not specified.

:file_name: backend_wrappers.py
:author: Jingchu Liu
:data: 2017-09-05
"""

from core import DrivingSimulatorEnv


class DrivingSimulatorEnvHonda(DrivingSimulatorEnv):
    """Fills backend commands if not specified."""
    def __init__(self, *args, **kwargs):
        if not 'backend_cmds' in kwargs or kwargs['backend_cmds'] is None:
            print "[__init__()]: using default backend cmds."
            ws_path = '/Projects/catkin_ws/'
            initialD_path = '/Projects/hobotrl/playground/initialD/'
            backend_path = initialD_path + 'ros_environments/backend_scripts/'
            utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
            kwargs['backend_cmds'] = [
                # 1. Parse maps
                ['python', utils_path+'parse_map.py',
                 ws_path+'src/Map/src/map_api/data/honda_wider.xodr',
                 utils_path+'road_segment_info.txt'],
                # 2. Generate obs and launch file
                ['python', utils_path+'gen_launch_dynamic.py',
                 utils_path+'road_segment_info.txt', ws_path,
                 utils_path+'honda_dynamic_obs_template.launch', 30],
                # 3. start roscore
                ['roscore'],
                # 4. start reward function script
                ['python', backend_path+'gazebo_rl_reward.py'],
                # 5. start simulation restarter backend
                ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
                # 6. [optional] video capture
                # ['python', backend_path+'non_stop_data_capture.py', 0]
            ]
        super(DrivingSimulatorEnvHonda, self).__init__(*args, **kwargs)


class DrivingSimulatorEnvGTA(DrivingSimulatorEnv):
    def __init__(self, *args, **kwargs):
        if not 'backend_cmds' in kwargs or kwargs['backend_cmds'] is None:
            print "[__init__()]: using default backend cmds."
            ws_path = '/Projects/catkin_ws/'
            initialD_path = '/Projects/hobotrl/playground/initialD/'
            backend_path = initialD_path + 'ros_environments/backend_scripts/'
            utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
            kwargs['backend_cmds'] = [
                # 1. start roscore
                ['roscore'],
                # 2. start reward function script
                ['python', backend_path+'gazebo_rl_reward.py'],
                # 3. start simulation restarter backend
                ['python', backend_path+'gta5_restart.py',
                 '--ip', '10.31.40.215',
                 '--port_number',' 10000',
                  '1'],
                # 4. [optional] video capture
                # ['python', backend_path+'non_stop_data_capture.py', 0]
            ]
        super(DrivingSimulatorEnvGTA, self).__init__(*args, **kwargs)


