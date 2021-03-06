# -*- coding: utf-8 -*-

"""Abstract base classes of RL agent components.
The base classes implement some common features of RL agent components,
and leave features unique to individual algorithms as abstract methods.

Date   : 2017-06-02
"""

from utils import Stepper, ScheduledParamCollector


class Agent(object):
    def __init__(self, *args, **kwargs):
        pass

    def step(self, state, action, reward, next_state,
             episode_done=False, **kwargs):
        pass

    def act(self, state, **kwargs):
        pass

    def new_episode(self, state):
        """
        called when a new episode starts.
        :param state:
        :return:
        """
        pass


class BaseAgent(Agent):
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

    def __init__(self, *args, **kwargs):
        super(BaseAgent, self).__init__(*args, **kwargs)
        self._stepper = Stepper(0)
        self._params = ScheduledParamCollector(*args, **kwargs)
        self._params.set_int_handle(self._stepper)
        pass

    @property
    def step_n(self):
        return self._stepper.value()

    def step(self, state, action, reward, next_state,
             episode_done=False, **kwargs):
        """Single Agent Step
        called when single interaction step with outside world occurs.
        The agent receive an experience tuple ("state", "action",
        "reward", "next_state") from the outside world and returns an action and
        relevant info.

        Optionally the outside world provides a "episode_done" argument to
        indicate the end of an interaction episode.

        Parameters
        ----------
        state  : state of outside world.
        action : action taken
        reward : scalar reward signal.
        next_state :afterstate
        episode_done : true if episode ends in this step.
        kwargs : other params
        """
        if not (state is None or action is None or reward is None or
                next_state is None or episode_done is None):
            self._stepper.step()
        learning_off = kwargs['learning_off'] if 'learning_off' in kwargs else False
        # Agent improve itself with new experience
        if not learning_off:
            info = self.reinforce_(
                state, action, reward, next_state,
                episode_done=episode_done, **kwargs)
        else:
            info = {}
        params = self._params.get_params()
        info.update(params)
        info.update(self._params.get_numeric_params())
        return info

    def new_episode(self, state):
        """
        called when a new episode starts.
        :param state:
        :return:
        """
        pass

    def act(self, state, **kwargs):
        """
        called when an action need to be taken from this agent.
        :param state:
        :param kwargs:
        :return:
        """
        raise NotImplementedError(
            "BaseAgent.act(): abstract method not implemented."
        )

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        # Default agent does nothing. This method is not
        #   abstract to facilitate super() call from
        #   child classes.
        return {}


class Policy(object):
    """Base class for policies.
    A Policy acts according to state.
    """
    def act(self, state, **kwargs):
        """Return an action according to incoming state.

        :param state:
        :return: action for current state
        """
        raise NotImplementedError()
