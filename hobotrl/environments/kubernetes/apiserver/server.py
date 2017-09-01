#
# -*- coding: utf-8 -*-

import logging
import json

import web
from control import KubeUtil, EnvTracker, EvictThread


urls = (
    "/spawn", "Spawn",
    "/stop/(.+)", "Stop",
    "/ping/(.+)", "Ping",
    "/attach/(.+)", "Attach"
)

app = web.application(urls, globals())

kube = KubeUtil(incluster=False)
env_tracker = EnvTracker(kube)


class Spawn(object):
    def GET(self):
        return json.dumps(kube.spawn_new_env())


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