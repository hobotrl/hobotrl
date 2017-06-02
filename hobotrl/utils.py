from core import *

from numpy import max
from numpy.random import rand, randint


class EpsilonGreedyPolicyMixin(BasePolicyMixin):
    """Epsilon greedy policy
    This policy superimpose a random policy onto the greedy policy with a small
    probability epsilon.

    Assume super-class already initialized a discrete and index-able action
    space. And assume a action-value func. mixin already implements the
    "get_value()" method for value retrieval.
    """
    def __init__(self, epsilon, tol=1e-10, **kwargs):
        """Initialization

        Parameters
        ----------
        epsilon : probability of choosing random action.
        tol     : a small tolerance for equality tests.
        """
        super(EpsilonGreedyPolicyMixin, self).__init__(**kwargs)

        self.EPSILON = epsilon
        self.TOL = tol

    def act(self, state, evaluate=False, **kwargs):
        """Epsilon greedy action selection.
        Choose greedy action with 1-epsilon probability and random action with
        epsilon probability. Ties are broken randomly for greedy actions.
        """
        if state is None or rand() < self.EPSILON:
            idx_action = randint(0, len(self.ACTIONS))
        else:
            # Follow greedy policy with 1-epsilon prob.
            # break tie randomly
            q_vals = self.get_value(state=state)
            max_q_val = max(q_vals)
            idx_best_actions = [
                i for i in range(len(q_vals))
                if (q_vals[i] - max_q_val)**2 < self.TOL
            ]
            idx_action = idx_best_actions[randint(0, len(idx_best_actions))]

        return self.ACTIONS[idx_action]


