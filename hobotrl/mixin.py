# -*- coding: utf-8 -*-

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

import numpy as np

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

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):
        parent_info = super(
            BaseValueMixin, self
        ).reinforce_(
            state, action, reward, next_state, episode_done=episode_done, **kwargs)

        eval_info = self.improve_value_(
            state, action, reward, next_state, episode_done=episode_done, **kwargs
        )
        parent_info.update(eval_info)
        return parent_info

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

    def reinforce_(self, state, action, reward, next_state, episode_done=False,
                   **kwargs):
        parent_info = super(BasePolicyMixin, self).reinforce_(
            state, action, reward, next_state, episode_done=episode_done,
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

    def act(self, *args, **kwargs):
        return self.__epgp.act(*args, **kwargs)


class ReplayMixin(object):
    """Experience Replay Wrapper
    This is a wrapper class that provides batch of uncorrelated experiences
    for the "reinforce_()" method of its parent class(es). It can be seen
    as a "translater": single sequential data in, batch uncorrelated data out.

    Experience replay is a method for preparing un-correlated experiences
    for RL algorithms. It uses a buffer to store (possibliy correlated)
    past experience and break the correlation through random sampling.

    Overriding Hierachy
    -------------------
    __init__:
        self.__init__
            |- [call super] XXX.__init__
    reinforce_:
        self.reinforce_
            |- [call super] XXX.reinforce_
    """
    def __init__(self, buffer_class, buffer_param_dict,
                 batch_size, f_prepare_sample=None,
                 **kwargs):
        """Initialization
        Since the parent class most probably will also use the "batch_size"
        argument (e.g. for building NN DAG), we repack it back into the kwargs
        before making the super call.

        Parameters
        ----------
        buffer_class : the class of the replay memory (not instance).
        buffer_param_dict : kwargs for initializating the memory.
        batch_size :
        f_prepare_sample : method to prepare sample from experience. Use
                           default if None is passed in.
        """
        kwargs['batch_size'] = batch_size  # super-class may need this info
        super(ReplayMixin, self).__init__(**kwargs)

        self.__BATCH_SIZE = batch_size
        self.__replay_buffer = buffer_class(**buffer_param_dict)
        if f_prepare_sample is None:
            self.prepare_sample = self.__default_prepare_sample
        else:
            self.prepare_sample = f_prepare_sample

    # TODO: For cases in which buffer needs to be updated, maybe we can
    #       set up a `update_buffer` method to update related fields
    #       using the reinforce_ info dict passed back.
    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        """Buffer update, sample, and supercall
        Push current experience into replay buffer. Samples a batch of
        experience and provide it to the `reinforce_()` method of parent
        classes.
        """
        # Update buffer
        self.__replay_buffer.push_sample(
            self.prepare_sample(state, action, reward, next_state,
                                episode_done, **kwargs)
        )

        # Super call
        info = super(ReplayMixin, self).reinforce_(state, action, reward, next_state, episode_done, **kwargs)

        return info

    def __default_prepare_sample(self, state, action, reward, next_state,
                       episode_done, **kwargs):
        """Adapt experience format
        Adapt the format of incoming experience to that of the `push_sample()`
        method. The "SARS" quadraple is the default format. This method can be
        can be overriden if other formats are needed.
        """

        sample = {
          "state": np.array(state),
          "action": np.array(action),
          "reward": np.array(reward),
          "next_state": np.array(next_state),
          "episode_done": np.array(episode_done)
        }
        return sample

    def reset_memory(self):
        """Reset the replay memory
        """
        self.__replay_buffer.reset()

    # TODO: user @property decorator?
    def get_replay_buffer(self):
        """
        get replay buffer
        :return:
        """
        return self.__replay_buffer

