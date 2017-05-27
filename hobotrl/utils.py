from core import *

class TabularQMixin(BaseValueFuncMixin):
    """
    TODO: DocString
    """
    def __init__(self, actions, gamma, on_policy=False, alpha=0.0,
                 default_q_val=0.0, **kwargs):
        super(TabularQMixin, self).__init__(**kwargs)

        self.ACTIONS = actions
        self.GAMMA = gamma  # discount factor
        self.ON_POLICY = on_policy
        self.ALPHA = alpha  # Moving average exponent for T-D updates
        self.DEFAULT_QVAL = default_q_val

        self.__q = {}

    def get_value(self, state, action=None, **kwargs):
        if action is None:  # return Q values on all actions
            return [
                self.__q[(state, a)] if (state, a) in self.__q else self.DEFAULT_QVAL
                for a in self.ACTIONS
            ]
        else:
            exp = (state, action)
            return self.__q[exp] if exp in self__q else self.DEFAULT_QVAL

    def improve_value_(self, state, action, reward, next_state,
                       episode_done=False, **kwargs):
        # Getting the "next_q":
        #   If on-policy evaluation, either use the "next_action" passed in
        #   or sample next action.
        #   If off-policy evaluation, uses the maximum Q value across all
        #   'next_action'.        
        next_q = self.get_value(
                     next_state,
                     kwargs['next_action'] if 'next_action' in kwargs \
                         else self.act_(next_state)
                 ) if self.ON_POLICY else max(self.get_value(next_state))

        # Target Q value from Bellman iteration
        target_q = reward + self.GAMMA * next_q * (1 - episode_done)

        # Standard Temporal Difference update with exponention moving averaging
        exp = (state, action)
        self.__q[exp] = self.ALPHA * target_q + \
            (1 - self.ALPHA) * (self.__q[exp] if exp in self.__q else self.DEFAULT_QVAL)

        return None

