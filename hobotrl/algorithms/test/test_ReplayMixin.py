import sys
sys.path.append('../')

from core import *
from utils import *
from playback import *

TestAgent(ReplayMixin, BaseAgent):
    def __init__(self, **kwargs):
        super(TestAgent, self).__init__(**kwargs)

replay = TestAgent(
            memory_class=MapPlayback,
            memory_param_dict={
                "capacity": 16,
                "sample_shapes": {
                    'last_state': (3,),
                    'last_action': (2,),
                    'reward': (1,),
                    'state': (3,)
                }
            },
            batch_size=4
        )


if __name__ == '__main__':


