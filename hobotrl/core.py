"""Abstract base classes of RL agent components.
The base classes implement some common features of RL agent components,
and leave features unique to individual algorithms as abstract methods.

Date   : 2017-06-02
"""

class BaseAgent(object):
    """Base class for reinforcement learning agents.
    This is the base class for general RL agents.

    The main interface with outside world is the step() function. In each
    step, the agent is fed with "last_state", "last_action", "state" and
    "reward" (and optionally "episode_done" in episodic environments). The
    agent will return an action and related info.

    Under the hood, the agent improves itself in the reinforce_() method
    using these input arguments. And the returned action is determined by
    the act() method.

    Note: semantically, the reinforce_() method hosts stateful
    procedures of the agent, e.g. value iteration, policy evaluation,
    policy improvement, etc, while the act() method may only host the
    stateless procedures, e.g. value lookup, probabilistic action selection,
    etc.

    Note: the act() methods is abstract and need to be implemented.
    """

    def __init__(self):
        pass

    def step(self, last_state, last_action, state, reward,
             episode_done=False, **kwargs):
        """Single interaction step with outside world.
        The agent receive an experience tuple ("last_state", "last_action",
        "state", "reward") from the outside world and returns an action and
        relevante info.

        Optionally the outside world provides a "episode_done" argument to
        indicate the end of an interaction episode.

        Parameters:
        state  : state of outside world.
        reward : scalar reward signal.
        episode_done : whether the interaction ends in this step.
        kwargs :
        """

        # Agent improve itself with new experience
        info = self.reinforce_(last_state, last_action, reward, state,
                               episode_done=False, **kwargs)

        # Agent take action in reaction to current state
        action = self.act(state, **kwargs)

        return action, info

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        # Default agent does nothing. This method is not
        #   abstract to fascilitate super() call from
        #   child classes.
        pass

    def act(self, state, **kwargs):
        raise NotImplementedError(
            "BaseAgent.act(): abstract method not implemented."
        )


class BaseValueFuncMixin(object):
    """Base class for value function mixins.
    This is the base class for the value function modules of
    value-based agents.

    The "reinforce_()" method first escalate call to parent
    class, and then improves the quality of the value function.
    Update info from parent class and value func. estimation is
    combined and returned.

    The abstract method "get_value()" should return action values
    given state and optionally action.

    The abstract method "improve_value_()" is supposed to improve
    the quality of value func. estimations.
    """
    def __init__(self, **kwargs):
        super(BaseValueFuncMixin, self).__init__(**kwargs)

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):

        parent_info = super(
            BaseValueFuncMixin, self
        ).reinforce_(
            state, action, reward, next_state,
            episode_done, **kwargs)

        eval_info = self.improve_value_(
            state, action, reward, next_state,
            episode_done, **kwargs
        )

        return parent_info, eval_info

    def get_value(self, state, action=None, **kwargs):
        raise NotImplementedError(
            "BaseValueFuncMixin.get_value() :" +
            "abstract method not implemented."
        )

    def improve_value_(self, state, action, reward, next_state,
                       episode_done, **kwargs):
        raise NotImplementedError(
            "BaseValueFuncMixin.improve_value_() :" +
            "abstract method not implemented."
        )


class BasePolicyMixin(object):
    """Base class for policy mixins.
    This is the base class for the policy of an RL agent (i.e.
    how to act). Materialized child class should at least
    implement the "act()" abstract method.
    """
    def __init__(self, **kwargs):
        super(BasePolicyMixin, self).__init__(**kwargs)

    def reinforce_(self, last_state, last_action, state ,reward,
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


