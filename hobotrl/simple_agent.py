from core import *
from utils import *

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


