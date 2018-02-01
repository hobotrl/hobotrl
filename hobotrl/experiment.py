#
# -*- coding: utf-8 -*-

import itertools
import logging
import os
import random
import sys
import traceback
from pathos.multiprocessing import ProcessingPool as Pool

import tensorflow as tf
import argparse

from utils import clone_params, escape_path

logging.basicConfig(format='[%(asctime)s] (%(filename)s): |%(levelname)s| %(message)s')


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

    def __init__(self, *args, **kwargs):
        super(HelloWorld, self).__init__()

    def run(self, args):
        """
        hello world
        :return:
        """
        x = 0
        for i in range(100000000):
            x = x + i
        logging.warning("hello experiment!")
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
        for parameter in GridSearch.product(self._parameters):
            label = GridSearch.labelize(parameter)
            args.logdir = GridSearch.find_new(os.sep.join([log_root, label]))
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

    @staticmethod
    def product(parameters):
        if isinstance(parameters, dict):
            parameters = [parameters]
        for param in parameters:
            names = sorted(param.keys())
            valuelists = [param[n] for n in names]
            for values in itertools.product(*valuelists):
                yield clone_params(**dict(zip(names, values)))

    @staticmethod
    def find_new(path):
        if not os.path.exists(path):
            return path
        for i in range(10000):
            ipath = "%s_%d" % (path, i)
            if not os.path.exists(ipath):
                return ipath
        return path

    @staticmethod
    def labelize(parameter):
        names = sorted(parameter.keys())
        return "_".join(["%s%s" % (f, escape_path(str(parameter[f]))) for f in names])


def subprocess_run(exp_key, log_root, parameter, label, args):
    sub_stdout, sub_stderr = None, None
    org_std = None

    def reset_log_handler():
        n_h = len(logging.root.handlers)
        for i in range(n_h):
            logging.root.removeHandler(logging.root.handlers[n_h - i - 1])
        logging.basicConfig(format="[%(asctime)s] %(message)s")
    try:
        args.logdir = GridSearch.find_new(os.sep.join([log_root, label]))
        os.makedirs(args.logdir)
        org_std = sys.stdout, sys.stderr
        sub_stdout = open(os.sep.join([args.logdir, "stdout.txt"]), "w")
        sub_stderr = open(os.sep.join([args.logdir, "stderr.txt"]), "w")
        logging.warning("starting task with logdir:%s", args.logdir)
        sys.stdout, sys.stderr = sub_stdout, sub_stderr
        reset_log_handler()
        # logging again to start in log file
        logging.warning("starting task with logdir:%s", args.logdir)
        with tf.Graph().as_default():
            exp_class = ParallelGridSearch.class_cache[exp_key]
            experiment = exp_class(**parameter)
            rewards = experiment.run(args)
    except Exception, e:
        type_, value_, traceback_ = sys.exc_info()
        traceback_ = traceback.format_tb(traceback_)
        traceback_ = "\n".join(traceback_)
        logging.warning("experiment[%s] failed:%s, %s, %s", label, type_, value_, traceback_)
    finally:
        if org_std is not None:
            sys.stdout, sys.stderr = org_std
        try:
            if sub_stdout is not None:
                sub_stdout.flush()
                sub_stdout.close()
                sub_stdout = None
        except:
            pass
        try:
            if sub_stderr is not None:
                sub_stderr.flush()
                sub_stderr.close()
                sub_stderr = None
        except:
            pass
        reset_log_handler()


class ParallelGridSearch(Experiment):

    class_cache = {}

    def __init__(self, exp_class, parameters, parallel=4):
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
        super(ParallelGridSearch, self).__init__()
        self._exp_class, self._parameters, self._parallel = exp_class, parameters, parallel
        self._key = random.random()
        ParallelGridSearch.class_cache[self._key] = exp_class
        logging.warning("cache content:%s %s", id(ParallelGridSearch.class_cache), ParallelGridSearch.class_cache)

    def run(self, args):
        self.pool = Pool(self._parallel)
        self.log_root = args.logdir
        parameters = list(GridSearch.product(self._parameters))
        labels = [GridSearch.labelize(p) for p in parameters]
        n = len(labels)
        logging.warning("total searched combination:%s", n)
        ret = self.pool.amap(subprocess_run,
                             [self._key] * n, [self.log_root] * n, parameters, labels, [args] * n)
        ret.wait()
        self.pool.close()
        self.pool.join()


if __name__ == "__main__":
    Experiment.main()
