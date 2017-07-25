import numpy as np
import math


class CrossEntropyMethodParameterGenerator(object):
    def __init__(self, parameter_shapes, n, proportion, initial_variance, noise=0.):
        assert n >= 1
        assert 0. < proportion < 1.
        assert math.floor(n*proportion) >= 1
        assert noise >= 0.
        assert initial_variance >= 0.

        self.para_shapes = parameter_shapes
        self.n = n
        self.proportion = proportion
        self.noise = noise

        self.means = [np.zeros(shape) for shape in parameter_shapes]
        self.variances = [np.zeros(shape) + initial_variance
                          for shape in parameter_shapes]

    def generate_parameter_list(self):
        return [np.random.normal(loc=self.means[i],
                                 scale=self.variances[i],
                                 size=self.para_shapes[i])
                for i in range(len(self.para_shapes))]

    def generate_parameter_lists(self):
        return [self.generate_parameter_list() for _ in range(self.n)]

    def update_parameter_lists(self, parameter_lists, scores):
        assert len(parameter_lists) == self.n
        assert len(scores) == self.n

        # Select parameters with top scores
        n_selected = int(math.floor(self.n*self.proportion))
        selected = sorted(range(len(parameter_lists)),
                          key=lambda x: scores[x],
                          reverse=True)[:n_selected]

        # Update mean value
        para_sum = [np.zeros(shape) for shape in self.para_shapes]
        for para_id in selected:
            parameters = parameter_lists[para_id]
            for i in range(len(para_sum)):
                para_sum[i] += parameters[i]
        self.means = [para/n_selected for para in para_sum]

        # Update variance
        squared_errors = [np.zeros(shape) for shape in self.para_shapes]
        for para_id in selected:
            parameters = parameter_lists[para_id]
            for i in range(len(squared_errors)):
                squared_errors[i] += (parameters[i] - self.means[i])**2
        self.variances = [squared_error/n_selected + self.noise for squared_error in squared_errors]

        # Generate new parameters
        for para_id in range(len(parameter_lists)):
            if para_id not in selected:
                parameter_lists[para_id] = self.generate_parameter_list()


def test():
    def f(x):
        return math.exp(-(x-2)**2) + 0.8*math.exp(-(x+2)**2)

    cem = CrossEntropyMethodParameterGenerator(parameter_shapes=[(1,)],
                                               n=10,
                                               proportion=0.5,
                                               initial_variance=50,
                                               noise=0.2)
    parameter_lists = cem.generate_parameter_lists()
    for i in range(1000):
        scores = [f(x[0]) for x in parameter_lists]
        cem.update_parameter_lists(parameter_lists, scores)

    print [float(para[0]) for para in parameter_lists]
    print "mean:"
    print float(cem.means[0])
    print "variance:"
    print float(cem.variances[0])
    print "score:"
    print scores
    print ""

test()
