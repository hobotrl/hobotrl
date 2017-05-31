from core import *

from numpy import max
from numpy.random import rand, randint

class TabularQMixin(BaseValueFuncMixin):
    """Table-based Action-Value Function.
    This class implements the classical table-based action value function
    (i.e. Q function). The Q values are stored in a dictionary for each
    (state, action) pair and can be updated with the temporal-difference (TD)
    learning algorithm.
    """
    def __init__(self, actions, gamma, greedy_policy=True, alpha=1.0,
                 default_q_val=0.0, **kwargs):
        """Initialization

        Parameters
        ----------
        actions : the action space. (Tuple)
        gamma   : discount factor for value functions. (Float)
        greedy_policy : whether or not to evaluate the greedy policy. (Bool)
        alpha   : (optional) 1-alpha is the exponential decay factor for old
                  Q values. (Float)
        default_q_val : (optional) default value for uninitialized action-value
                        entries. (Float)
        """
        super(TabularQMixin, self).__init__(**kwargs)

        self.ACTIONS = actions
        self.GAMMA = gamma  # discount factor
        self.GREEDY_POLICY = greedy_policy
        self.ALPHA = alpha  # Moving average exponent for T-D updates
        self.DEFAULT_QVAL = default_q_val

        self.__q = {}

    def get_value(self, state, action=None, **kwargs):
        """Retrieve action-value entries
        Return action-value entry for specified (state, action) pair or for all
        actions of a particular state if the "action" is None.

        Parameters
        ----------
        state  :
        action :
        """
        if action is None:  # return Q values for all actions
            return [
                self.__q[(state, a)] if (state, a) in self.__q else \
                self.DEFAULT_QVAL
                for a in self.ACTIONS
            ]
        else:
            exp = (state, action)
            return self.__q[exp] if exp in self__q else self.DEFAULT_QVAL

    def improve_value_(self, state, action, reward,
                       next_state, next_action=None,
                       episode_done=False, importance=1.0,
                       **kwargs):
        """Evaluate policy with one-step temporal difference.
        This method evaluate a policy by means of the temporal difference
        algorithm and forms a tabular action-value function.

        Depending on the class attr. "GREEDY_POLICY", this method either
        evaluate the greedy policy (True) or evaluate other policies (False).
        In the latter case, the "importance" arg. can also be provided for
        off-policy evaluation. It will be used to correct the bias on action
        selection.

        Note the "importance" arg. will be ignored in the greedy policy case
        as well as in the on-policy case with default next action.

        Parameters
        ----------
        state  :
        action :
        reward :
        next_state   :
        next_action  :
        episode_done :
        importance   : importance sampling ratio for off-policy evaluation.
                       Use default (1.0) for greedy of on-policy evaluation.
        """
        # Getting the Q value for next step:
        # If evaluate the greedy policy use the maximum Q value across all
        # actions.
        if self.GREEDY_POLICY:
            # greedy policy suggests an unit importance
            importance = 1.0
            next_q = max(self.get_value(next_state))
        # If evaluate other policies, either use the "next_action" passed in
        # or sample next action with "act_()" if "next_action" is None.
        else:
            # on-policy with default next_action suggest an unit importance 
            importance = 1.0 if next_action is None else importance
            next_action = self.act_(next_state) if next_action is None else next_action
            next_q = self.get_value(next_state, next_action)

        # Target Q value from Bellman iteration
        target_q = reward + self.GAMMA * importance * next_q * (1 - episode_done)

        # Standard Temporal Difference update with exponention moving
        # averaging, i.e update is the average of old value and new target.
        exp = (state, action)
        if exp not in self.__q:
            self.__q[exp] = self.DEFAULT_QVAL
        td = target_q - self.__q[exp]
        self.__q[exp] += self.ALPHA * td

        return td

    def reset(self, **kwargs):
        """Reset Action Value Table
        """
        super(TabularQMixin, self).reset(**kwargs)
        self.__q = {}


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

    def act_(self, state, **kwargs):
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


