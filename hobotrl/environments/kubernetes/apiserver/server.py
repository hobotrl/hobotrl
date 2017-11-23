#
# -*- coding: utf-8 -*-

import logging
import json

import web
from control import KubeUtil, EnvTracker, EvictThread


urls = (
    "/spawn/(.+)", "Spawn",
    "/spawn", "Spawn",
    "/stop/(.+)", "Stop",
    "/ping/(.+)", "Ping",
    "/attach/(.+)", "Attach"
)

app = web.application(urls, globals(), autoreload=False)

kube = KubeUtil(incluster=False)
env_tracker = EnvTracker(kube)


class Spawn(object):
    DEFAULT_IMAGE = "docker.hobot.cc/carsim/simulator_cpu_kub:0.0.6"

    def GET(self, image_uri=None):
        if image_uri is None:
            logging.warning("using default:%s", self.DEFAULT_IMAGE)
            image_uri = self.DEFAULT_IMAGE
        return json.dumps(kube.spawn_new_env(image_uri))


class Stop(object):
    def GET(self, env_id):
        env_tracker.terminate(env_id)
        return json.dumps({"result": "ok"})


class Ping(object):
    """
    the client should ping this server, presumably at interval shorter than a minute, as keep-alive signal.
    Server would evict
    """
    def GET(self, env_id):
        env_tracker.keepalive(env_id)
        return json.dumps({"result": "ok"})

if __name__ == '__main__':
    evictor = EvictThread(env_tracker)
    evictor.start()
    app.run()  # endless loop
    evictor.stop()