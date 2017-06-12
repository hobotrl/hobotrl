import sys
sys.path.append('../')

import numpy as np
from mixin import BasePolicyMixin, OUExplorationMixin


class EchoBase(BasePolicyMixin):
    def __init__(self, **kwargs):
        pass

    def act(self, state, **kwargs):
        return state

class TestOU(OUExplorationMixin, EchoBase):
    def __init__(self, **kwargs):
        super(TestOU, self).__init__(**kwargs)

# Agent initialization
print "========================="
print "Check OU initialization: "
ou_params=(0.1, 0.05, 0.01)
action_shape=(2,)

test_ou = TestOU(ou_params=ou_params, action_shape=action_shape)
print "pass!"

# State evolution
print "=========================="
print "Check OU state evolution (action_i = state_i): "
state = np.zeros(action_shape)
print "Initial ou state is {}".format(test_ou.ou_state.flatten())
print "Initial env state is {}".format(state.flatten())
for n in range(50):
    state = test_ou.act(state)
    print "OU state at step {} is {}".format(n , test_ou.ou_state.flatten()),
    print "Env state is {}".format(state.flatten())
print "pass!"
