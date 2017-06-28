from gym import spaces
import gym
import numpy as np
import random

# Layer ID for each type of object
FOOD = 0
BODY = 1
HEAD = 2
ONES = 3  # A layer of all ones

N_LAYER = 4  # Number of layers


class SnakeGame(gym.core.Env):
    action_space = spaces.Discrete(4)  # Action space
    metadata = {'render.modes': ['ansi']}

    offset_for_direction = [(0,-1), (0,1),
                            (-1, 0), (1,0)]  # Position offset for moving up, down, left and right

    def __init__(self, size_x, size_y, head_x, head_y, food_reward=1., wall_reward=0., n_food=1):
        """
        The classic Snake game.

        :param size_x(int): map width
        :param size_y(int): map height
        :param head_x(int): x coordinate of the snake's initial position
        :param head_y(int): y coordinate of the snake's initial position
        :param food_reward(float): reward for getting food
        :param wall_reward(float): reward for hitting the border or the snake itself
        :param n_food(int): number of foods in the map
        """
        assert size_x > 0 and size_y > 0
        assert 0 <= head_x < size_x and 0 <= head_y < size_y
        assert n_food >= 1

        self.done = None  # Whether episode is done
        self.frame_count = None  # Number of frames since last reset

        self.n_food = n_food

        self.size_x = size_x
        self.size_y = size_y
        self.init_head_x = head_x
        self.init_head_y = head_y
        self.food_reward = food_reward
        self.wall_reward = wall_reward

        # Reset the environment
        self.state = self.body_direction = \
            self.head_x = self.head_y = self.tail_x = self.tail_y \
            = None
        self.reset()

        self.observation_space = spaces.Box(0., 1., shape=np.array(self.state).shape)

    def generate_food(self):
        """
        Generate a new food on the map if there's enough space.

        :return: whether there's enough space on the map.
        """
        # Check whether there's enough space
        if len(self.body_direction) + self.n_food >= self.size_x*self.size_y:
            return False

        # Randomly generate food position
        while True:
            food_x = random.randrange(self.size_x)
            food_y = random.randrange(self.size_y)

            # Try again if this point is occupied
            if self.state[food_x][food_y][BODY] == 1 \
                    or self.state[food_x][food_y][FOOD] == 1:
                continue
            break

        self.state[food_x][food_y][FOOD] = 1

        return True

    def _reset(self):
        """
        Reset the environment.
        """
        # Reset the environment state
        self.state = [[[0]*N_LAYER for i in range(self.size_x)] for j in range(self.size_y)]

        for i in range(self.size_x):
            for j in range(self.size_y):
                self.state[i][j][ONES] = 1

        # Reset the snake
        self.body_direction = []
        self.head_x = self.tail_x = self.init_head_x
        self.head_y = self.tail_y = self.init_head_y
        self.state[self.head_x][self.head_y][BODY] = 1
        self.state[self.head_x][self.head_y][HEAD] = 1

        # Reset foods
        for i in range(self.n_food):
            self.generate_food()

        # Reset counter
        self.frame_count = 0
        self.done = False

        return np.array(self.state)

    def _render(self, mode='ansi', close=False):
        """
        Render the environment in console.
        """
        if close:
            return

        assert mode == 'ansi'

        result = ''

        def str_for_grid(x, y):
            """
            Get the string representation for a grid.

            :param x: location of the grid
            :param y: location of the grid
            :return(str): string representation
            """
            if not (0 <= x < self.size_x and 0 <= y < self.size_y):
                return "##"

            if self.state[x][y][FOOD] == 1:
                return "* "

            if self.state[x][y][BODY] == 1:
                if self.state[x][y][HEAD] == 1:
                    return "{}"
                else:
                    return "[]"
            else:
                return "  "

        # Enumerate the grids including the border
        for y in range(-1, self.size_y + 1):
            for x in range(-1, self.size_x + 1):
                result += str_for_grid(x, y)
            result += '\n'

        return result

    def _step(self, action):
        """
        Take an action.

        :param action: ID of the action
        :return: a tuple (state, reward, episode_done, info)
        """
        reward = 0.0  # Reward in this step

        dx, dy = self.offset_for_direction[action]  # Position offset of this action

        # Calculate next position of the snake's head
        next_position_x = self.head_x + dx
        next_position_y = self.head_y + dy

        # Check whether the snake hits the border or itself
        if not (0 <= next_position_x < self.size_x and 0 <= next_position_y < self.size_y) \
                or self.state[next_position_x][next_position_y][BODY] == 1:
            # Episode done
            self.done = True
            reward += self.wall_reward
        else:
            # Move the head forward
            self.state[self.head_x][self.head_y][HEAD] = 0

            self.head_x = next_position_x
            self.head_y = next_position_y

            self.state[self.head_x][self.head_y][HEAD] \
                = self.state[self.head_x][self.head_y][BODY] = 1

            # Save the snake's shape
            self.body_direction.insert(0, action)

            # Check whether there is a food
            if self.state[next_position_x][next_position_y][FOOD] == 1:
                # Clean the food and generate a new one
                self.state[next_position_x][next_position_y][FOOD] = 0
                reward += self.food_reward
                self.generate_food()
            else:
                # Move the tail forward
                dx, dy = self.offset_for_direction[self.body_direction[-1]]
                self.body_direction.pop()

                self.state[self.tail_x][self.tail_y][BODY] = 0
                self.tail_x += dx
                self.tail_y += dy

        self.frame_count += 1

        return np.array(self.state), reward, self.done, {}


def test():
    import time
    frame_time = 0.5

    env = SnakeGame(5, 5, 2, 2)

    print env.render('ansi')
    time.sleep(frame_time)

    while True:
        result = env.step(env.action_space.sample())
        print result
        print env.render('ansi')

        if result[2]:
            print env.reset()

        time.sleep(frame_time)

if __name__ == "__main__":
    test()
