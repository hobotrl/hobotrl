import zmq
import dill

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

