#
# -*- coding: utf-8 -*-

import logging
import threading
import time
import random
import yaml
import kubernetes as kube
from monotonic import monotonic_time


class KubeUtil(object):
    """
    KubeUtil talks to kubernetes api server.
    """
    namespace = "simulator"
    SPAWN_RETRY = 10
    POD_WAIT = 100
    SVC_WAIT = 100
    ENV_PREFIX = "ros-env-"

    def __init__(self, incluster=False):
        super(KubeUtil, self).__init__()
        if incluster:
            kube.config.load_incluster_config()
        else:
            kube.config.load_kube_config()
        self.api = kube.client.CoreV1Api()

    def get_svc_list(self):
        return self.api.list_namespaced_service(self.namespace)

    def get_pod_list(self):
        """
        retrieve list of envs
        :return:
        """
        return self.api.list_namespaced_pod(self.namespace)

    def destroy_env(self, env_id):
        """
        terminate an env
        :param env_id:
        :return:
        """
        try:
            self.api.delete_namespaced_pod(env_id, self.namespace, body={})
            logging.warning("pod destroyed: %s", env_id)
        except:
            pass
        try:
            self.api.delete_namespaced_service(env_id, self.namespace)
            logging.warning("svc destroyed: %s", env_id)
        except:
            pass

    def spawn_new_env(self, image_uri):
        """
        create new pod and service accessible from outside kubernetes cluster
        :return: dict{"id": id, "host": host, "port": port} to access newly spawned env
        :rtype: dict
        """
        for i in range(self.SPAWN_RETRY):
            pod_object, svc_object = None, None
            try:
                env_id = self.ensure_env_id_()
                pod = yaml.load(open("pod.yaml").read().replace("${id}", env_id).replace("${image}", image_uri))
                pod_object = self.api.create_namespaced_pod(namespace=self.namespace, body=pod)
                # wait for pod deploy ok
                pod_host = None
                for i in range(self.POD_WAIT):
                    pod_object = self.api.read_namespaced_pod(env_id, self.namespace)
                    pod_host = pod_object.status.host_ip
                    if pod_host is not None:
                        break
                    time.sleep(1)
                if pod_host is None:
                    # cannot retrieve host ip, deploy pod consider failed
                    raise IOError("cannot determine pod_host after %d retry" % self.POD_WAIT)
                svc = yaml.load(open("svc.yaml").read().replace("${id}", env_id))
                svc_object = self.api.create_namespaced_service(self.namespace, svc)
                # wait for svc deploy ok
                svc_port = {}
                for i in range(self.SVC_WAIT):
                    svc_object = self.api.read_namespaced_service(env_id, self.namespace)
                    if svc_object.spec is not None and svc_object.spec.ports is not None:
                        for port in svc_object.spec.ports:
                            svc_port[port.name] = port.node_port
                    # svc_port = svc_object.spec.ports[0].node_port \
                    #     if svc_object.spec is not None and svc_object.spec.ports is not None and \
                    #     len(svc_object.spec.ports) >= 1 \
                    #     else None
                    if len(svc_port) > 0:
                        break
                    time.sleep(1)
                if len(svc_port) == 0:
                    # cannot retrieve service port, deploy svc consider failed
                    raise IOError("cannot determine svc_port after %d retry" % self.SVC_WAIT)
                env = {"id": env_id, "host": pod_host, "port": svc_port}
                logging.warning("env created: %s", env)
                return env
            except RuntimeError, e:
                if pod_object is not None:
                    self.api.delete_namespaced_pod(env_id, self.namespace, body={})
                if svc_object is not None:
                    self.api.delete_namespaced_service(env_id, self.namespace)

    def gen_env_id_(self):
        return self.ENV_PREFIX + str(random.randint(0, 1e10))

    def ensure_env_id_(self):
        for i in range(10):
            env_id = self.gen_env_id_()
            pods = self.api.list_namespaced_pod(self.namespace)
            names = [item.metadata.name for item in pods.items]
            if env_id not in names:
                return env_id
        raise ValueError("cannot create env_id! %s" % env_id)


class EnvTracker(object):

    TIMEOUT = 300  # 5 minutes before considered dead

    """
    tracks keep-alive signals for all env; evicts old envs from time to time
    """
    def __init__(self, kube_util):
        """
        :param kube_util:
        :type kube_util: KubeUtil
        """
        super(EnvTracker, self).__init__()
        self.keep_alives = {}  # map from env_id to last keep-alive timestamp
        self.kube = kube_util

    def keepalive(self, env_id):
        self.keep_alives[env_id] = monotonic_time()

    def terminate(self, env_id):
        if env_id in self.keep_alives:
            self.keep_alives.pop(env_id)
        self.kube.destroy_env(env_id)

    def evict(self):
        envs = self.kube.get_svc_list()
        env_ids = filter(lambda n: n.startswith(self.kube.ENV_PREFIX),
                         [e.metadata.name for e in envs.items])
        now = monotonic_time()
        logging.warning("eviction start:%s", env_ids)

        # terminate outdated envs
        for env_id in env_ids:
            if env_id not in self.keep_alives:
                self.keep_alives[env_id] = now  # ensure every env_id appears in keep_alives at least once
            elif now - self.keep_alives[env_id] > self.TIMEOUT:
                # deadline passed
                logging.warning("timeout for %s: %s", env_id, (now - self.keep_alives[env_id]))
                self.kube.destroy_env(env_id)

        # remove desolated keepalives
        for env_id in self.keep_alives.keys():
            if env_id not in env_ids:
                self.keep_alives.pop(env_id)


class EvictThread(threading.Thread):

    INTERVAL = 30  # seconds

    def __init__(self, env_tracker, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        if name is None:
            name = "EvictThread"
        super(EvictThread, self).__init__(group, target, name, args, kwargs, verbose)
        self.env_tracker = env_tracker
        self.stopped = False
        self.daemon = True
        self._sleeper = threading.Event()

    def run(self):
        while not self.stopped:
            self._sleeper.wait(self.INTERVAL)
            self.env_tracker.evict()

    def stop(self):
        self.stopped = True
        self._sleeper.set()
