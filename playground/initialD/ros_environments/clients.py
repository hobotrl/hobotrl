import zmq
import dill
import numpy as np
import hobotrl as hrl


class DrivingSimulatorEnvClient(object):
    def __init__(self, address, port, **kwargs):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(address, port))
        kwargs['func_compile_obs'] = dill.dumps(
            kwargs['func_compile_obs'])
        kwargs['func_compile_reward'] = dill.dumps(
            kwargs['func_compile_reward'])
        self.socket.send_pyobj(('start', kwargs))
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
        msg_type, msg_payload = self.socket.recv_pyobj()
        self.socket.close()
        self.context.term()
        if not msg_type == 'exit':
            raise Exception('EnvClient: msg_type is not exit.')
        return


class Driving1Env(DrivingSimulatorEnvClient):
    def __init__(self, address, port, **kwargs):
        def compile_obs(obss):
            obs1 = obss[-1][0]
            print obss[-1][1]
            print obs1.shape
            return obs1

        # Environment
        def compile_reward(rewards):
            return rewards

        def compile_reward_agent(rewards):
            global momentum_ped
            global momentum_opp
            rewards = np.mean(np.array(rewards), axis=0)
            print (' ' * 10 + 'R: [' + '{:4.2f} ' * len(rewards) + ']').format(*rewards),

            # obstacle
            rewards[0] *= -100.0
            # distance to
            rewards[1] *= -1.0 * (rewards[1] > 2.0)
            # velocity
            rewards[2] *= 10
            # opposite
            momentum_opp = (rewards[3] < 0.5) * (momentum_opp + (1 - rewards[3]))
            momentum_opp = min(momentum_opp, 20)
            rewards[3] = -20 * (0.9 + 0.1 * momentum_opp) * (momentum_opp > 1.0)
            # ped
            momentum_ped = (rewards[4] > 0.5) * (momentum_ped + rewards[4])
            momentum_ped = min(momentum_ped, 12)
            rewards[4] = -40 * (0.9 + 0.1 * momentum_ped) * (momentum_ped > 1.0)

            reward = np.sum(rewards) / 100.0
            print '{:6.4f}, {:6.4f}'.format(momentum_opp, momentum_ped),
            print ': {:7.4f}'.format(reward)
            return reward

        super(Driving1Env, self).__init__(address, port,
            defs_obs=[
                ('/training/image/compressed', 'sensor_msgs.msg.CompressedImage'),
                ('/decision_result', 'std_msgs.msg.Int16')
            ],
            func_compile_obs=compile_obs,
            defs_reward=[
                ('/rl/has_obstacle_nearby', 'std_msgs.msg.Bool'),
                ('/rl/distance_to_longestpath', 'std_msgs.msg.Float32'),
                ('/rl/car_velocity', 'std_msgs.msg.Float32'),
                ('/rl/last_on_opposite_path', 'std_msgs.msg.Int16'),
                ('/rl/on_pedestrian', 'std_msgs.msg.Bool')],
            func_compile_reward=compile_reward,
            defs_action=[('/autoDrive_KeyboardMode', 'std_msgs.msg.Char')],
            rate_action=10.0,
            window_sizes={'obs': 2, 'reward': 3},
            buffer_sizes={'obs': 2, 'reward': 3},
            step_delay_target=0.5)


class KubernetesDriving1Env(hrl.envs.KubernetesEnv):
    def __init__(self, api_server_address="train078.hogpu.cc:30794"):
        super(KubernetesDriving1Env, self).__init__(Driving1Env, api_server_address)