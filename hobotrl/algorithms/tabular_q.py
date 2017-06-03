# -*- coding: utf-8 -*-

from hobotrl.core import BaseAgent
from hobotrl.mixin import EpsilonGreedyPolicyMixin, TabularQMixin

class TabularQLearning(
    EpsilonGreedyPolicyMixin,
    TabularQMixin,
    BaseAgent):
    """Q-Learning Agent.
    Canonical tablular Q learning agent.
    
    Overriding Hierachy
    -------------------
    __init__:
        self.__init__
        |- [call super] EpsilonGreedyPolicyMixin.__init__
            |- [call super] BasePolicyMixin.__init__
                |- [call super] TabularQMixin.__init__
                    |- [call super] BaseValueMixin.__init__
                        |- [call super] BaseAgent.__init__
    reinforce_:
        BasePolicyMixin.reinforce_
        |- [call super] BaseValueMixin.reinforce_
            |- [call super] BaseAgent.reinforce_

    act:
        EpsilonGreedyPolicyMixin.act
        |- [call member] EpsilonGreedyPolicy.act
        |- [override] BasePolicyMixin.act
            |- [override] BaseAgent.act
    """
    def __init__(self, **kwargs):
        """
        """
        # force evaluate greedy policy
        kwargs['greedy_policy'] = True
        
        super(TabularQLearning, self).__init__(**kwargs)
        






