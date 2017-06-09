import sys
sys.path.append('../')

from core import BaseAgent
import numpy as np
from mixin import ReplayMixin
from playback import MapPlayback

class ReplayAgent(ReplayMixin, BaseAgent):
    def __init__(self, **kwargs):
        super(ReplayAgent, self).__init__(**kwargs)

# Basic push and sample functionality
print "============================="
print "Check basic functionalities: ",
agent = ReplayAgent(
            buffer_class=MapPlayback,
            buffer_param_dict={
                "capacity": 16,
                "sample_shapes": {
                    'state': (3,),
                    'action': (2,),
                    'reward': (1,),
                    'next_state': (3,),
                    'episode_done': (1,)
                }
            },
            batch_size=4
        )

for i in range(32):
    agent.reinforce_(
        state=(i,i,i), action=(i, i), reward=i, next_state=(i,i,i)
    )
    print agent._ReplayMixin__replay_buffer.sample_batch(1)
print "Pass!"

# Default `prepare_sample()`
print "============================================="
print "Check if `prepare_sample` is set as default: ",
assert agent.prepare_sample == agent._ReplayMixin__default_prepare_sample
print "Pass!"

# Specified prepare_sample()
print "==============================================="
print "Check if we can pass in custom `prepare_sample`"

def f_prepare_sample(state, action, reward, next_state,
                       episode_done, **kwargs):
    sample = {
        "state": np.array(state),
        "action": np.array(action),
        "reward": np.array(reward),
        "next_state": np.array(next_state),
        "episode_done": np.array(episode_done),
        "test_field": np.random.rand(2,)
    }
    return sample

agent = ReplayAgent(
            buffer_class=MapPlayback,
            buffer_param_dict={
                "capacity": 16,
                "sample_shapes": {
                    'state': (3,),
                    'action': (2,),
                    'reward': (1,),
                    'next_state': (3,),
                    'episode_done': (1,),
                    'test_field': (2,)
                }
            },
            batch_size=4,
            f_prepare_sample=f_prepare_sample
        )
assert agent.prepare_sample == f_prepare_sample

for i in range(32):
    agent.reinforce_(
        state=(i,i,i), action=(i, i), reward=i, next_state=(i,i,i)
    )
    print agent._ReplayMixin__replay_buffer.sample_batch(1)
print "pass!"

