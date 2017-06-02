#
# -*- coding: utf-8 -*-

from hobotrl.core import BaseValueFuncMixin
from hobotrl.core import BaseAgent
from hobotrl.utils import EpsilonGreedyPolicyMixin


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
            return self.__q[exp] if exp in self.__q else self.DEFAULT_QVAL

    def improve_value_(self, state, action, next_state, reward,
                       next_action=None,
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
        # or sample next action with "act()" if "next_action" is None.
        else:
            # on-policy with default next_action suggest an unit importance
            importance = 1.0 if next_action is None else importance
            next_action = self.act(next_state) if next_action is None else next_action
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

    # TODO: seems this method has no practical use anywhere
    # def reset(self, **kwargs):
    #     """Reset Action Value Table
    #     """
    #     super(TabularQMixin, self).reset(**kwargs)
    #     self.__q = {}


class TabularQLearning(
    EpsilonGreedyPolicyMixin,
    TabularQMixin,
    BaseAgent):
    """Q-Learning Agent.
    Canonical tablular Q learning agent.
    """
    def __init__(self, **kwargs):
        kwargs['greedy_policy'] = True  # force evaluate greedy policy
        super(TabularQLearning, self).__init__(**kwargs)


