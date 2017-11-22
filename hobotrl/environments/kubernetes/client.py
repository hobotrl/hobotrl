#
# -*- coding: utf-8 -*-


import logging
import threading
import Queue
import httplib2
import urllib
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
    def __init__(self, remote_client_env_class,
                 api_server_address="train078.hogpu.cc:30794",
                 image_uri=None,
                 **kwargs):
        """

        :param remote_client_env_class: remote environment class, with constructor signature:
            __init__(self, address, port, **kwargs)

        :param api_server_address: caller should pass api server address to overwrite
                default server address.
        """
        super(KubernetesEnv, self).__init__()
        self._remote_env_class = remote_client_env_class
        self._api_server_address = api_server_address

        self._cls_kwargs = kwargs
        self._env, self._env_spec = None, None
        self._init_env_queue = Queue.Queue(maxsize=1)
        self._api_thread = ApiThread(remote_client_env_class, kwargs, api_server_address, image_uri, self._init_env_queue)
        self._api_thread.start()

    def reset(self):
        if self._env is None:
            self._env_spec = self._init_env_queue.get()
            if isinstance(self._env_spec, RuntimeError):
                logging.warning("environment creation failed")
                raise self._env
            elif isinstance(self._env_spec, dict):
                self._env = self._env_spec["env"]
                del self._env_spec["env"]
            else:
                self._env = self._env_spec
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def exit(self):
        result = self._env.exit()
        self._api_thread.stop()
        return result


class ApiThread(threading.Thread):

    PING_INTERVAL = 60

    def __init__(self, remote_client_env_class, cls_kwargs, api_server_address, image_uri, env_queue,
                 group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ApiThread, self).__init__(group, target, name, args, kwargs, verbose)
        self._env_queue = env_queue
        self._remote_env_class = remote_client_env_class
        self._cls_kwargs = cls_kwargs
        self._api_server_address = "http://" + api_server_address if api_server_address.find("http://") !=0 \
            else api_server_address
        self._image_uri = image_uri
        self._stopped = False
        self._sleeper = threading.Event()

    def run(self):
        # init env
        try:
            http = httplib2.Http()
            if self._image_uri is not None:
                url = self._api_server_address + "/spawn/" + self._image_uri
            else:
                url = self._api_server_address + "/spawn"
            response = http.request(url)[1]
            env_spec = json.loads(response)
            logging.warning("remote env spec:%s", env_spec)
            env_id = env_spec["id"]
            port = env_spec["port"]
            if isinstance(port, dict):
                # multiple port available; choose websocket port
                port = port["websocket"]
            env_spec["env"] = self._remote_env_class(env_spec["host"], port, **self._cls_kwargs)
            self._env_queue.put(env_spec)
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
