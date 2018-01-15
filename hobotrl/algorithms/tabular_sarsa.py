# -*- coding: utf-8 -*-

from hobotrl.core import BaseAgent
from tabular_q import TabularQFunc, TabularQLearning

class SARSA(
    TabularQLearning):
    """SARSA On-Policy Learning

    improve_value_:
        self.improve_value_
        |- [decorates] TabularQ.improve_value_
            |- [call member] TabularQFunc.improve_value
    """
    def __init__(self, num_action, discount_factor=0.9, epsilon_greedy=0.2, **kwargs):
        """
        """
        # force evaluate behavioral policy
        kwargs['greedy_policy'] = False
        
        super(SARSA, self).__init__(num_action, discount_factor, epsilon_greedy, **kwargs)
        
        # Ensure behavioral policy is available through `act()`
        try:
            dummy = self.act
        except:
            raise ValueError(
                'SARSA: '
                'method `act()` not properly initialized.'
            )
                
        # decorator style argument enforcement
        self.improve_value_ = self.__on_policy(self.improve_value_)       
    
    def __on_policy(self, f_improve_value):
        """Decorator for Enforcing On-Policy Evaluation
        Enforce on-policy evaluation by setting arguments
        overriding `next_action` and `importance`.
        
        Assume 'next_action' is 5th in position, 'importance'
        is 7th in position.
        """
        def f_improve_value_on(*args, **kwargs):
            args = list(args)
            next_state = args[3] if len(args)>3 else kwargs['next_state']
            next_action = self.act(next_state)  # sample behavioral policy
            # enforcing next action
            if len(args) > 4:
                args[4] = next_action
            else:
                kwargs['next_action'] = next_action
            # enforcing importance
            if len(args) > 6:
                args[6] = 1.0
            else:
                kwargs['importance'] = 1.0
            return f_improve_value(*args, **kwargs)
        
        return f_improve_value_on
