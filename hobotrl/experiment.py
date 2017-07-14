#
# -*- coding: utf-8 -*-

import tensorflow as tf
import gym
import argparse


class Experiment(object):
    """
    """
    experiments = {}

    def __init__(self):
        self.desc = ""
        pass

    def run(self, args):
        pass

    @staticmethod
    def register(subclass, desc=None):
        e = subclass()
        if desc is not None:
            e.desc = desc
        name = subclass.__name__
        if Experiment.experiments.has_key(name):
            raise KeyError("experiment: %s already exists!" % name)
        Experiment.experiments[name] = e

    @staticmethod
    def start(name, args):
        if not args.logdir:
            args.logdir = args.name
        Experiment.experiments[name].run(args)
        pass

    @staticmethod
    def list():
        msg = ""
        for k in Experiment.experiments:
            v = Experiment.experiments[k]
            msg += str(k) + ":\t" + str(v.desc) + "\n"
        return msg

    @staticmethod
    def main():
        import os
        parser = argparse.ArgumentParser()
        parser.add_argument("operation", default="list")
        parser.add_argument("--name", default="HelloWorld")
        parser.add_argument("--logdir", default=None)
        parser.add_argument("--job", default="worker")
        parser.add_argument("--index", default="0")
        parser.add_argument("--cluster",
                            default="{'ps':['localhost:2222'], " \
                                    "'worker':['localhost:2223', 'localhost:2224', 'localhost:2225']}")
        args = parser.parse_args()
        # args.logdir = os.path.join("log", args.name if args.logdir is None else args.logdir)
        args.logdir = os.path.join("log", args.name) if args.logdir is None else args.logdir
        args.index = int(args.index)
        if args.operation == "list":
            print Experiment.list()
        elif args.operation == "run":
            Experiment.start(args.name, args)


class HelloWorld(Experiment):
    def run(self, args):
        """
        hello world
        :return:
        """
        print "hello experiment!"
Experiment.register(HelloWorld, "first experiment")


if __name__ == "__main__":
    Experiment.main()
