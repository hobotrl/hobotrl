#
# -*- coding: utf-8 -*-

import sys
import logging
import tensorflow as tf
import gym
import argparse
import os

from utils import clone_params_dict, escape_path


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
        experiment_class().run(args)
        logging.warning("Experiment %s end.", name)
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

        """
        super(GridSearch, self).__init__()
        self._exp_class, self._parameters = exp_class, parameters

    def run(self, args):
        log_root = args.logdir
        for parameter in self.product(self._parameters):
            args.logdir = os.sep.join([log_root, self.labelize(parameter)])
            with tf.Graph().as_default():
                experiment = self._exp_class(**parameter)
                experiment.run(args)

    def product(self, parameters):
        counts = dict([(k, len(parameters[k])) for k in parameters])
        current = [[k, 0] for k in parameters]
        while True:
            parameter = dict([(k[0], parameters[k[0]][k[1]]) for k in current])
            yield clone_params_dict(**parameter)
            # to next
            has_next = False
            for i in range(len(current)-1, -1, -1):
                field = current[i]
                if field[1] < len(parameters[field[0]]) - 1:
                    field[1] += 1
                    has_next = True
                    break
                else:
                    field[1] = 0
            if not has_next:
                break

    def labelize(self, parameter):
        return "_".join(["%s%s" % (f, escape_path(str(parameter[f]))) for f in parameter])


if __name__ == "__main__":
    Experiment.main()
