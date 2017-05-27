"""Abstract base classes of RL agent components.
The base classes implement some common features of RL agent components,
and leave features unique to individual algorithms as abstract methods.

Date   : 2017-05-27
"""

class BaseAgent(object):
    """Base class for reinforcement learning agents.
    This is the base class for general RL agents.

    The main interface with outside world is the step() function. In each
    step, the agent is fed with "state" and "reward" (and optionally
    "episode_done" in episodic environments. The agent will return an
    action and related info.

    Under the hood, the agent improves itself in the reinforce_() method
    using these input arguments as well as "state" and "action" buffered
    in the last step. And the returned action is determined by the
    act_() method.

    Note: the act_() methods is abstract and need to be implemented.
    """

    def __init__(self):
        self.__last_state = None
        self.__last_action = None

    def step(self, state, reward, episode_done=False, **kwargs):
        """Single interaction step with outside world.
        The agent receive the "state" of outside world and a scalar
        "reward" from past interaction and returns a action and
        relevante info.

        Optionally the outside world provides a "episode_done"
        signal to indicate the end of an interaction episode.

        Parameters:
        state  : state of outside world.
        reward : scalar reward signal.
        episode_done : whether the interaction ends in this step.
        kwargs :
        """

        # Agent improve itself with new experience
        info = self.reinforce_(
            self.__last_state, self.__last_action,
            reward, state, episode_done=False, **kwargs
        )

        # Agent take action in reaction to current state
        action = self.act_(state, **kwargs)
        self.__last_state = state
        self.__last_action = action

        return action, info

    def reset(self):
        self.__last_state = None
        self.__last_action = None

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        # Default agent does nothing. This method is not
        #   abstract to fascilitate super() call from
        #   child classes.
        pass

    def act_(self, state, **kwargs):
        raise NotImplementedError(
            "BaseAgent.act_(): abstract method not implemented."
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
    implement the "act_()" abstract method.
    """
    def __init__(self, **kwargs):
        pass

    def act_(self, state, **kwargs):
        raise NotImplementedError(
            "BasePolicyMixin.act_() :" +
            "abstract method not implemented."
        )


