#
# -*- coding: utf-8 -*-

import logging

import web
from control import KubeUtil, EnvTracker, EvictThread


urls = (
    "/spawn", "Spawn",
    "/ping", "Ping",
    "/attach", "Attach"
)

app = web.application(urls, globals())

kube = KubeUtil(incluster=False)
env_tracker = EnvTracker(kube)


class Spawn(object):

    def GET(self):
        return kube.spawn_new_env()


class Ping(object):
    """
    the client should ping this server, presumably at interval shorter than a minute, as keep-alive signal.
    Server would evict
    """
    def GET(self, env_id):
        env_tracker.keepalive(env_id)
        return {"result": "ok"}

if __name__ == '__main__':
    evictor = EvictThread(env_tracker)
    evictor.start()
    app.run()  # endless loop
    evictor.stop()