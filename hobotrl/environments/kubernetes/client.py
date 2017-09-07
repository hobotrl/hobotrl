#
# -*- coding: utf-8 -*-


import logging
import threading
import Queue
import httplib2
import json


class RemoteEnvClient(object):
    def __init__(self, address, port, **kwargs):
        super(RemoteEnvClient, self).__init__()

    def reset(self):
        pass

    def step(self, action):
        pass

    def exit(self):
        pass


class KubernetesEnv(object):
    def __init__(self, remote_env_class, api_server_address="train078.hogpu.cc:30794"):
        """

        :param remote_env_class: remote environment class, with constructor signature:
            __init__(self, address, port, **kwargs)

        :param api_server_address: caller should pass api server address to overwrite
                default server address.
        """
        super(KubernetesEnv, self).__init__()
        self._remote_env_class = remote_env_class
        self._api_server_address = api_server_address
        self._env = None
        self._init_env_queue = Queue.Queue(maxsize=1)
        self._api_thread = ApiThread(remote_env_class, api_server_address, self._init_env_queue)
        self._api_thread.start()

    def reset(self):
        if self._env is None:
            self._env = self._init_env_queue.get()
            if isinstance(self._env, RuntimeError):
                logging.warning("environment creation failed")
                raise self._env
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def exit(self):
        result = self._env.exit()
        self._api_thread.stop()
        return result


class ApiThread(threading.Thread):

    PING_INTERVAL = 60

    def __init__(self, remote_env_class, api_server_address, env_queue,
                 group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ApiThread, self).__init__(group, target, name, args, kwargs, verbose)
        self._env_queue = env_queue
        self._remote_env_class = remote_env_class
        self._api_server_address = "http://" + api_server_address if api_server_address.find("http://") !=0 \
            else api_server_address
        self._stopped = False
        self._sleeper = threading.Event()

    def run(self):
        # init env
        try:
            http = httplib2.Http()
            response = http.request(self._api_server_address + "/spawn")[1]
            env_spec = json.loads(response)
            env_id = env_spec["id"]
            self._env_queue.put(self._remote_env_class(env_spec["host"], env_spec["port"]))
        except RuntimeError, e:
            logging.warning("error creating env: %s", e)
            self._env_queue.put(e)
            self._stopped = True
            return
        # keep alive
        while not self._stopped:
            http.request(self._api_server_address + "/ping/" + env_id)
            self._sleeper.wait(self.PING_INTERVAL)
        # terminate environment
        http.request(self._api_server_address + "/stop/" + env_id)

    def stop(self):
        self._stopped = True
        self._sleeper.set()
