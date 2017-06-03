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

    def __init__(self, **kwargs):
        pass

    def step(self, last_state, last_action, reward, state,
             episode_done=False, **kwargs):
        """Single interaction step with outside world.
        The agent receive an experience tuple ("last_state", "last_action",
        "reward", "state") from the outside world and returns an action and
        relevante info.

        Optionally the outside world provides a "episode_done" argument to
        indicate the end of an interaction episode.

        Parameters
        ----------
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

