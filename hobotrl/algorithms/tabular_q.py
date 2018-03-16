# -*- coding: utf-8 -*-

import logging
import numpy as np

from hobotrl.core import BaseAgent
from hobotrl.policy import EpsilonGreedyPolicy
from hobotrl.network import Function


class TabularQFunc(Function):
    """Table-based Action-Value Function.
    This class implements the classical table-based action value function
    (i.e. Q function). The Q values are stored in a dictionary for each
    (state, action) pair and can be updated with the temporal-difference (TD)
    learning algorithm.
    """
    def __init__(self, actions, gamma, greedy_policy=True, alpha=0.1,
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
        self.__ACTIONS = actions
        print "__ACTIONS:", actions, type(self.__ACTIONS)
        self.__GAMMA = gamma  # discount factor
        self.__GREEDY_POLICY = greedy_policy
        self.__ALPHA = alpha  # Moving average exponent for T-D updates
        self.__DEFAULT_QVAL = default_q_val

        self.__q = {}

    def __call__(self, *args, **kwargs):
        # support batch access
        state = args[0]
        return [self.get_value(s) for s in state]

    def normalize_state(self, state):
        if type(state) == np.ndarray or type(state) == list:
            state = tuple(state)
        return state

    def get_value(self, state, action=None, **kwargs):
        """Retrieve action-value entries
        Return action-value entry for specified (state, action) pair or for all
        actions of a particular state if the "action" is None.

        Parameters
        ----------
        state  :
        action :
        """
        state = self.normalize_state(state)
        if action is None:  # return Q values for all actions
            return [
                self.__q[(state, a)] if (state, a) in self.__q else \
                self.__DEFAULT_QVAL
                for a in self.__ACTIONS
            ]
        else:
            exp = (state, action)
            return self.__q[exp] if exp in self.__q else self.__DEFAULT_QVAL

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
        state = self.normalize_state(state)
        if self.__GREEDY_POLICY:
            # greedy policy suggests an unit importance
            importance = 1.0
            next_q = max(self.get_value(next_state))
        # If evaluate other policies, either use the "next_action" passed in
        # or sample next action with "act_()" if "next_action" is None.
        else:
            next_q = self.get_value(next_state, next_action)

        # Target Q value from Bellman iteration
        target_q = reward + self.__GAMMA * importance * next_q * (1 - episode_done)

        # Standard Temporal Difference update with exponention moving
        # averaging, i.e update is the average of old value and new target.
        exp = (state, action)
        if exp not in self.__q:
            self.__q[exp] = self.__DEFAULT_QVAL
        td = target_q - self.__q[exp]
        self.__q[exp] += self.__ALPHA * td
        return {'td': td}


class TabularQLearning(
    BaseAgent):
    """Q-Learning Agent.
    Canonical tablular Q learning agent.
    
    """
    def __init__(self, num_action, discount_factor=0.9, epsilon_greedy=0.2, **kwargs):
        """
        """
        # force evaluate greedy policy
        kwargs['greedy_policy'] = True
        super(TabularQLearning, self).__init__(**kwargs)
        self.__tqf = TabularQFunc(actions=list(range(num_action)),
                                  gamma=discount_factor)
        self.get_value = \
            lambda *args, **kwargs: self.__tqf.get_value(*args, **kwargs)
        self.improve_value_ = \
            lambda *args, **kwargs: self.__tqf.improve_value_(*args, **kwargs)
        self.policy = EpsilonGreedyPolicy(self.__tqf, epsilon_greedy, num_action)

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        super(TabularQLearning, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)
        ret = self.improve_value_(state, action, reward, next_state, episode_done)
        return ret

    def act(self, state, **kwargs):
        action = self.policy.act(state)
        return action






