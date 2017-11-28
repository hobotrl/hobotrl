#
# -*- coding: utf-8 -*-


import logging
import threading
import Queue
import httplib2
import urllib
import json

import wrapt


class RemoteEnvClient(object):
    def __init__(self, address, port, **kwargs):
        super(RemoteEnvClient, self).__init__()

    def reset(self):
        pass

    def step(self, action):
        pass

    def exit(self):
        pass


class KubernetesEnv(wrapt.ObjectProxy):
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
        remote_env_class = remote_client_env_class
        api_server_address = "http://" + api_server_address if api_server_address.find("http://") !=0 \
            else api_server_address

        http = httplib2.Http()
        if image_uri is not None:
            url = api_server_address + "/spawn/" + image_uri
        else:
            url = api_server_address + "/spawn"

        response = http.request(url)[1]
        env_spec = json.loads(response)
        logging.warning("remote env spec:%s", env_spec)
        env_id = env_spec["id"]
        port = env_spec["port"]
        if isinstance(port, dict):
            # multiple port available; choose websocket port
            port = port["websocket"]
        env = remote_env_class(env_spec["host"], port, **kwargs)
        super(KubernetesEnv, self).__init__(env)
        self._env_spec = env_spec
        self._api_thread = ApiThread(api_server_address, env_id)
        self._api_thread.start()

    def exit(self):
        result = self.__wrapped__.exit()
        self._api_thread.stop()
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()


class ApiThread(threading.Thread):

    PING_INTERVAL = 60

    def __init__(self, api_server_address, env_id,
                 group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ApiThread, self).__init__(group, target, name, args, kwargs, verbose)
        self.daemon = True
        self._api_server_address = "http://" + api_server_address if api_server_address.find("http://") !=0 \
            else api_server_address
        self._env_id = env_id
        self._stopped = False
        self._sleeper = threading.Event()

    def run(self):
        http = httplib2.Http()
        # keep alive
        while not self._stopped:
            self._sleeper.wait(self.PING_INTERVAL)
            try:
                http.request(self._api_server_address + "/ping/" + self._env_id)
            except Exception, e:
                logging.warning("error when keeping alive:%s", e)
                pass
        # terminate environment
        http.request(self._api_server_address + "/stop/" + self._env_id)

    def stop(self):
        self._stopped = True
        self._sleeper.set()
