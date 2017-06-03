"""Mixin-style agents and utilities
This module provide mixins for blending components such as value
function and policy into an agent base class.

The `reinforce_()` method in BaseAgent will be overriden and turned
into a chain of super-calls per the Python Method Resolution Order
(a.k.a MRO). For example, if a agent is composed with the following
inheritance order:
    BaseAgent -> MixinA -> MixinB
Then the order of `reinforce_()` calls will actually be:
    MixinB.reinforce_() -> MixinA.reinforce_() -> BaseAgent.reinforce_()
Even if the `reinforce_()` method does nothing it should at least make
the super call to guarantee all `reinforce_()` methods along the chain
is called.

The abstract `act_()` method in BaseAgent is expect to be implemented
in policy mixin classes via proper overriding.

TODO: [Lewis] mixins with overriding are linear in nature, not sure if
      it will fit asyc. algorithms. So try best to keep common modules
      decoupled with mixins.
"""
from core import BaseAgent
from utils import TabularQFunc, EpsilonGreedyPolicy


class BaseValueMixin(object):
    """Base class for value function mixins.
    This is the base class for mixin-style value function modules
    of value-based agents.

    The `reinforce_()` method first escalate call to parent
    class, and then improves the quality of the value function.
    Update info from parent class and value func. estimation is
    combined and returned.

    The abstract method `get_value()` should return action values
    given state and optionally action.

    The abstract method `improve_value_()` is supposed to improve
    the quality of value func. estimations.
    """
    def __init__(self, **kwargs):
        super(BaseValueMixin, self).__init__(**kwargs)

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):

        parent_info = super(
            BaseValueMixin, self
        ).reinforce_(
            state, action, reward, next_state,
            episode_done=episode_done, **kwargs)

        eval_info = self.improve_value_(
            state, action, reward, next_state,
            episode_done=episode_done, **kwargs
        )

        return parent_info, eval_info

    def get_value(self, state, action=None, **kwargs):
        raise NotImplementedError(
            "BaseValueMixin.get_value() :" +
            "abstract method not implemented."
        )

    def improve_value_(self, state, action, reward, next_state,
                       episode_done, **kwargs):
        raise NotImplementedError(
            "BaseValueMixin.improve_value_() :" +
            "abstract method not implemented."
        )


class BasePolicyMixin(object):
    """Base class for policy mixins.
    This is the base class for the policy of an RL agent (i.e.
    how to act). Materialized child class should at least
    implement the `act()` abstract method.
    """
    def __init__(self, **kwargs):
        super(BasePolicyMixin, self).__init__(**kwargs)

    def reinforce_(self, last_state, last_action, state, reward,
                   **kwargs):
        parent_info = super(BasePolicyMixin, self).reinforce_(
            last_state, last_action, state, reward,
            **kwargs
        )

        return parent_info

    def act(self, state, **kwargs):
        raise NotImplementedError(
            "BasePolicyMixin.act() :" +
            "abstract method not implemented."
        )

        
class TabularQMixin(BaseValueMixin):
    """Thin mixin wrapper for Tabular Q Functions
    Make proper super call during `init_()` and implements
    `get_value()` and `improve_value()` via a very thin
    lambda wrapper over the member TabularQFunc instance.

    Overriding Hierachy
    -------------------
    __init__:
        self.__init__
    get_value:
        |- [call member] TabularQFunc.get_value
    improve_value:
        |- [call member] TabularQFunc.improve_value
    ``
    """
    def __init__(self, **kwargs):
        super(TabularQMixin, self).__init__(**kwargs)
        self.__tqf = TabularQFunc(**kwargs)
        self.get_value = \
            lambda *args, **kwargs: self.__tqf.get_value(*args, **kwargs)
        self.improve_value_ = \
            lambda *args, **kwargs: self.__tqf.improve_value_(*args, **kwargs)


class EpsilonGreedyPolicyMixin(BasePolicyMixin):
    """Thin Wrapper for EpsilonGreedyPolicy
    Make proper super call during `init_()` and implements `act()` via a
    very thin lambda wrapper over the member EpsilonGreedyPolicy instance.
    
    Requires super class to initialize a `get_value` method (most
    conviniently with a value mixin).
    
    Overriding Hierachy
    -------------------
    __init__
        self.__init__
        |- [call super] BasePolicyMixin.__init__
    act
        self.act
        |- [call member] EpsilonGreedyPolicy.act
        |- [override] BasePolicyMixin.act
    """
    def __init__(self, **kwargs):
        super(EpsilonGreedyPolicyMixin, self).__init__(**kwargs)
        
        # Check if `get_value` is properly initialized
        try:
            kwargs['f_get_value'] = self.get_value
        except:
            raise ValueError(
                'EpsilonGreedyPolicyMixin: '
                'method `get_value()` not properly initialized.'
            )

        self.__epgp = EpsilonGreedyPolicy(**kwargs)
        self.act = \
            lambda *args, **kwargs: self.__epgp.act(*args, **kwargs)


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
        

class SARSA(
    EpsilonGreedyPolicyMixin,
    TabularQMixin,
    BaseAgent):
    """SARSA On-Policy Learning

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
    
    improve_value_:
        self.improve_value_
        |- [decorates] TabularQ.improve_value_
            |- [call member] TabularQFunc.improve_value
    """
    def __init__(self, **kwargs):
        """
        """
        # force evaluate behavioral policy
        kwargs['greedy_policy'] = False
        
        super(SARSA, self).__init__(**kwargs)
        
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
        
        