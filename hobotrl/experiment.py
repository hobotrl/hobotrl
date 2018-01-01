#
# -*- coding: utf-8 -*-

import itertools
import logging
import os
import sys
import traceback

import tensorflow as tf
import gym
import argparse

from utils import clone_params, escape_path


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
        e = subclass
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
        experiment_class = Experiment.experiments[name]
        logging.warning("Experiment %s running with arg: %s", name, args)
        experiment_class().run(args)
        logging.warning("Experiment %s end.", name)
        pass

    @staticmethod
    def list():
        msg = ""
        for k in Experiment.experiments:
            v = Experiment.experiments[k]
            msg += "{:20}".format(str(k)) + ":\t" + str(v.desc) + "\n"
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
        parser.add_argument("--render_interval", default="-1")
        parser.add_argument("--render_once", default="true")
        parser.add_argument("--episode_n", default="1000")
        parser.add_argument("--cluster",
                            default="{'ps':['localhost:2222'], " \
                                    "'worker':['localhost:2223', 'localhost:2224', 'localhost:2225']}")
        args = parser.parse_args()
        # args.logdir = os.path.join("log", args.name if args.logdir is None else args.logdir)
        args.logdir = os.path.join("log", args.name) if args.logdir is None else args.logdir
        args.index = int(args.index)
        args.render_interval = int(args.render_interval)
        args.render_interval = sys.maxint if args.render_interval < 0 else args.render_interval
        args.render_once = args.render_once == 'true'
        args.episode_n = int(args.episode_n)
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


class GridSearch(Experiment):
    def __init__(self, exp_class, parameters):
        """
        :param exp_class: subclass of Experiment to run
        :type exp_class: class<Experiment>
        :param parameters: dict of list, experiment parameters to search within, i.e.:
            {
                "entropy": [1e-2, 1e-3],
                "learning_rate": [1e-3, 1e-4],
                ...
            }
            or list of dict-of-list, representing multiple groups of parameters:
            [
            {
                "entropy": [1e-2, 1e-3],
                "learning_rate": [1e-3, 1e-4],
                ...
            },
            {
                "batch_size": [32, 64],
                ...
            }
            ]

        """
        super(GridSearch, self).__init__()
        self._exp_class, self._parameters = exp_class, parameters

    def run(self, args):
        log_root = args.logdir
        for parameter in self.product(self._parameters):
            label = self.labelize(parameter)
            args.logdir = self.find_new(os.sep.join([log_root, label]))
            with tf.Graph().as_default():
                experiment = self._exp_class(**parameter)
                try:
                    logging.warning("starting experiment: %s", args.logdir)
                    rewards = experiment.run(args)
                except Exception, e:
                    type_, value_, traceback_ = sys.exc_info()
                    traceback_ = traceback.format_tb(traceback_)
                    traceback_ = "\n".join(traceback_)
                    logging.warning("experiment[%s] failed:%s, %s, %s", label, type_, value_, traceback_)

    def product(self, parameters):
        if isinstance(parameters, dict):
            parameters = [parameters]
        for param in parameters:
            names = sorted(param.keys())
            valuelists = [param[n] for n in names]
            for values in itertools.product(*valuelists):
                yield clone_params(**dict(zip(names, values)))

    def find_new(self, path):
        if not os.path.exists(path):
            return path
        for i in range(10000):
            ipath = "%s_%d" % (path, i)
            if not os.path.exists(ipath):
                return ipath
        return path

    def labelize(self, parameter):
        names = sorted(parameter.keys())
        return "_".join(["%s%s" % (f, escape_path(str(parameter[f]))) for f in names])


if __name__ == "__main__":
    Experiment.main()
