from numpy.random import randint


class GridworldSink:
    """A simple maze game with a single goal state
    This is a simple maze game environment. The game is played on a 2-D
    grid world and one of the grids is the goal grid. The agent can move up,
    down, left, and right in order to reach the goal grid, and the episode
    will end when the agent arrives at the goal grid.

    The grids by the boundary are fenced. The agent can choose to move
    towards the fence, but will remain there and (optionally) receive a
    "wall_reward".
    """
    def __init__(self, dims=None, goal_state=None,
                 goal_reward=100, wall_reward=0, null_reward=0):
        """
        Paramters
        ---------
        dims :
        goal_state  :
        goal_reward :
        wall_reward :
        null_reward :
        """
        self.ACTIONS = ['left', 'right', 'up', 'down']  # legitimate ACTIONS

        if dims is None:
            self.DIMS = (4, 5)
        else:
            self.DIMS = dims

        if goal_state is None:
            self.GOAL_STATE = (2, 2)
        else:
            self.GOAL_STATE = goal_state

        self.GOAL_REWARD = goal_reward
        self.WALL_REWARD = wall_reward
        self.NULL_REWARD = null_reward

        self.state = None
        self.done = False

        self.reset()

    def step(self, action):
        """

        Parameters
        ----------
        action : must be contailed in self.ACTIONS
        -------

        """
        if self.done:
            raise ValueError("Episode done, please restart.")
 
        next_state, reward, self.done  = self.transition_(self.state, action)
        self.state = next_state
        return next_state, reward, self.done, None

    def transition_(self, current_state, action):
        """State transition and rewarding logic

        """
        if action == 'up':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[0] == 0 \
                else ((current_state[0]-1, current_state[1]), self.NULL_REWARD)
        elif action == 'down':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[0] == (self.DIMS[0] - 1) \
                else ((current_state[0]+1, current_state[1]), self.NULL_REWARD)
        elif action == 'left':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[1] == 0 \
                else ((current_state[0], current_state[1]-1), self.NULL_REWARD)
        elif action == 'right':
            next_state, reward = (current_state, self.WALL_REWARD) if current_state[1] == (self.DIMS[1] - 1) \
                else ((current_state[0], current_state[1]+1), self.NULL_REWARD)
        else:
            print 'I don\'t understand this action ({}), I\'ll stay.'.format(action)
            next_state, reward = current_state, self.NULL_REWARD

        done = False
        if next_state == self.GOAL_STATE:
            reward = self.GOAL_REWARD
            done = True

        return next_state, reward, done

    def optimal_policy(self, state):
        if state[0] < self.GOAL_STATE[0]:
            return 'down'
        elif state[0] > self.GOAL_STATE[0]:
            return 'up'
        elif state[1] < self.GOAL_STATE[1]:
            return 'right'
        else:
            return 'left'

    def reset(self):
        """Randomly throw the agent to a non-goal state

        """
        next_state = self.GOAL_STATE
        while next_state == self.GOAL_STATE:
            next_state = (randint(0, self.DIMS[0]), randint(0, self.DIMS[1]))
        self.state = next_state
        self.done = False

        return self.state

    def isDone(self):
        return self.state == self.GOAL_STATE
