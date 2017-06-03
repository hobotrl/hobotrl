import sys
sys.path.append('../')

from core import BaseAgent
from mixin import ReplayMixin
from playback import MapPlayback

class ReplayAgent(ReplayMixin, BaseAgent):
    def __init__(self, **kwargs):
        super(ReplayAgent, self).__init__(**kwargs)

agent = ReplayAgent(
            buffer_class=MapPlayback,
            buffer_param_dict={
                "capacity": 16,
                "sample_shapes": {
                    'state': (3,),
                    'action': (2,),
                    'reward': (1,),
                    'next_state': (3,)
                }
            },
            batch_size=4
        )

for i in range(32):
    agent.reinforce_(
        state=(i,i,i), action=(i, i), reward=i, next_state=(i,i,i)
    )
    print agent._ReplayMixin__replay_buffer.pop_batch(1)

