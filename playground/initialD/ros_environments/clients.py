import sys
import zmq
import dill
import wrapt
import numpy as np
# HobotRL
sys.path.append('../../../')
from hobotrl.environments.kubernetes.client import KubernetesEnv
from server import DrSimDecisionK8SServer
# Gym
from gym.spaces import Discrete, Box


class DrivingSimulatorEnvClient(object):
    def __init__(self, address, port, **kwargs):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(address, port))
        kwargs_send = {}
        for key, value in kwargs.iteritems():
            kwargs_send[key] = dill.dumps(value)
        self.socket.send_pyobj(('start', kwargs_send))
        msg_type, msg_payload = self.socket.recv_pyobj()
        if not msg_type == 'start':
            raise Exception('EnvClient: msg_type is not start.')

    def reset(self):
        self.socket.send_pyobj(('reset', None))
        msg_type, msg_payload = self.socket.recv_pyobj()
        if not msg_type == 'reset':
            raise Exception('EnvClient: msg_type is not reset.')
        return msg_payload

    def step(self, action):
        self.socket.send_pyobj(('step', (action,)))
        msg_type, msg_payload = self.socket.recv_pyobj()
        if not msg_type == 'step':
            raise Exception('EnvClient: msg_type is not step.')
        return msg_payload

    def exit(self):
        self.socket.send_pyobj(('exit', None))
        self.socket.close()
        # self.context.term()
        return

    def close(self):
        self.exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()


class DrSimDecisionK8S(wrapt.ObjectProxy):
    _version = '20171127'
    _ALL_ACTIONS = [(ord(mode),) for mode in ['s', 'd', 'a']] + [(0,)]
    def __init__(self, image_uri=None, backend_cmds=None, *args, **kwargs):
        # Simulator Docker image to use
        if image_uri is None:
            _image_uri = "docker.hobot.cc/carsim/simulator_gpu_kub:latest"
        else:
            _image_uri = image_uri

        # bash commands to be executed in each episode to start simulation
        # backend
        if backend_cmds is None:
            if "launch" in kwargs:
                launch = kwargs["launch"]
            else:
                launch = None
            _backend_cmds = self.gen_default_backend_cmds(launch)
        else:
            _backend_cmds = backend_cmds

        # ROS topic definition tuples for observation, reward, and action
        _defs_obs = [
            ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
            ('/decision_result', 'std_msgs.msg.Int16'),
            ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
        ]
        _defs_reward = [
            ('/rl/car_velocity_front', 'std_msgs.msg.Float32'),
            ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
            ('/rl/obs_factor', 'std_msgs.msg.Float32'),
            ('/rl/current_road_validity', 'std_msgs.msg.Int16'),
            ('/rl/entering_intersection', 'std_msgs.msg.Bool'),
            ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
            ('/rl/on_biking_lane', 'std_msgs.msg.Bool'),
            ('/rl/on_innerest_lane', 'std_msgs.msg.Bool'),
            ('/rl/on_outterest_lane', 'std_msgs.msg.Bool')
        ]
        _defs_action = [('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')]

        _func_compile_obs = DrSimDecisionK8SServer.func_compile_obs
        _func_compile_reward = DrSimDecisionK8SServer.func_compile_reward
        _func_compile_action = DrSimDecisionK8SServer.func_compile_action

        # Build wrapped environment, expose step() an reset()
        _env = KubernetesEnv(
            image_uri=_image_uri,
            remote_client_env_class=DrivingSimulatorEnvClient,
            backend_cmds=_backend_cmds,
            defs_obs=_defs_obs,
            defs_reward=_defs_reward,
            defs_action=_defs_action,
            rate_action=10.0,
            window_sizes={'obs': 3, 'reward': 3},
            buffer_sizes={'obs': 3, 'reward': 3},
            func_compile_obs=_func_compile_obs,
            func_compile_reward=_func_compile_reward,
            func_compile_action=_func_compile_action,
            step_delay_target=0.5,
            **kwargs
        )
        super(DrSimDecisionK8S, self).__init__(_env)

        # Gym env required attributes
        self.observation_space = Box(low=0, high=255, shape=(350, 350, 3))
        self.reward_range = Box(
            low=-np.inf, high=np.inf, shape=(len(_defs_reward),)
        )
        self.action_space = Discrete(len(self._ALL_ACTIONS))
        self.metadata = {}

    @staticmethod
    def gen_default_backend_cmds(launch=None):
        ws_path = '/Projects/catkin_ws/'
        initialD_path = '/Projects/hobotrl/playground/initialD/'
        backend_path = initialD_path + 'ros_environments/backend_scripts/'
        utils_path = initialD_path + 'ros_environments/backend_scripts/utils/'
        if launch is None:
            launch = 'honda_dynamic_obs_template_tilt.launch'
        backend_cmds = [
            # Parse maps
            ['python', utils_path + 'parse_map.py',
             ws_path + 'src/Map/src/map_api/data/honda_wider.xodr',
             utils_path + 'road_segment_info.txt'],
            # Generate obstacle configuration and write to launch file
            ['python', utils_path+'gen_launch_dynamic_v1.py',
             utils_path+'road_segment_info.txt', ws_path,
             utils_path+launch, 32, '--random_n_obs'],
            # Start roscore
            ['roscore'],
            # Reward function script
            ['python', backend_path + 'gazebo_rl_reward.py'],
            # Road validity node script
            ['python', backend_path + 'road_validity.py',
             utils_path + 'road_segment_info.txt.signal'],
            # Simulation restarter backend
            ['python', backend_path+'rviz_restart.py', 'honda_dynamic_obs.launch'],
            # Video capture
            # ['python', backend_path+'non_stop_data_capture.py']
        ]
        return backend_cmds

    def exit(self):
        self.__wrapped__.exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()


class DrSimDecisionK8STopView(DrSimDecisionK8S):
    def __init__(self, image_uri=None, backend_cmds=None, *args, **kwargs):
        kwargs.update({
            "launch": "state_remap_test.launch"
        })
        super(DrSimDecisionK8STopView, self).__init__(image_uri, backend_cmds, *args, **kwargs)
