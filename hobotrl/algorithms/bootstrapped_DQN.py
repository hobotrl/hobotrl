import sys
sys.path.append(".")

import hobotrl as hrl
from environments import snake
import tensorflow as tf
import numpy as np
import random


def bernoulli_mask(p):
    return lambda n: [1 if random.random() < p else 0 for i in range(n)]


class BootstrappedDQN(hrl.tf_dependent.base.BaseDeepAgent):
    def __init__(self, observation_space, action_space,
                 nn_constructor, loss_function, trainer,
                 reward_decay, td_learning_rate, target_sync_interval,
                 replay_buffer_class, replay_buffer_args, min_buffer_size, batch_size=20,
                 n_heads=10, bootstrap_mask=bernoulli_mask(0.5)):
        assert 0. <= reward_decay <= 1.
        assert 0 < td_learning_rate <= 1.
        assert target_sync_interval > 0
        assert callable(nn_constructor)
        assert callable(loss_function)
        assert callable(trainer)
        assert issubclass(replay_buffer_class, hrl.playback.MapPlayback)
        assert n_heads > 0
        assert callable(bootstrap_mask)

        super(BootstrappedDQN, self).__init__(sess=tf.Session())

        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_decay = reward_decay
        self.td_learning_rate = td_learning_rate
        self.target_sync_interval = target_sync_interval
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.mask_generator = bootstrap_mask

        self.current_head = 0
        self.step_count = 0

        # Initialize replay buffer
        self.reply_buffer = replay_buffer_class(sample_shapes={
                                                    "state": observation_space.shape,
                                                    "next_state": observation_space.shape,
                                                    "action": [],
                                                    "reward": [],
                                                    "episode_done": []
                                                },
                                                **replay_buffer_args)

        # Construct the neural network
        with tf.variable_scope('non-target') as scope_non_target:
            nn = nn_constructor(observation_space=observation_space,
                                action_space=action_space,
                                n_heads=n_heads)
            assert len(nn["input"]) == self.n_heads
            assert len(nn["head"]) == self.n_heads

            self.nn_inputs = nn["input"]
            self.nn_heads = nn["head"]

        with tf.variable_scope('non-target') as scope_target:
            nn = nn_constructor(observation_space=observation_space,
                                action_space=action_space,
                                n_heads=n_heads)
            assert len(nn["input"]) == self.n_heads
            assert len(nn["head"]) == self.n_heads

            self.target_nn_inputs = nn["input"]
            self.target_nn_heads = nn["head"]

        # Construct synchronize operation
        non_target_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=scope_non_target.name)
        target_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=scope_target.name)
        self.op_sync = [tf.assign(target_var, non_target_var) for target_var, non_target_var
                        in zip(target_vars, non_target_vars)]

        # Construct loss function
        self.nn_outputs = [tf.placeholder(tf.float32, (None, action_space.n))
                           for _ in range(self.n_heads)]
        nn_output = tf.concat(self.nn_heads, 0)
        nn_target = tf.concat(self.nn_outputs, 0)
        self.loss = loss_function(output=nn_output, target=nn_target)
        self.train_op = trainer(self.loss)

        # Initialize the neural network
        self.get_session().run(tf.global_variables_initializer())
        self.get_session().run(self.op_sync)

    def act(self, state, **kwargs):
        action_values = self.get_session().run(self.nn_heads[self.current_head],
                             {self.nn_inputs[self.current_head]: [state]})
        print action_values
        return np.argmax(action_values)

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=False, **kwargs):
        self.reply_buffer.push_sample({
            "state": state,
            "next_state": next_state,
            "action": np.asarray(action),
            "reward": np.asarray(reward),
            "episode_done": np.asarray(episode_done)})

        # Randomly choose a head
        if episode_done:
            self.current_head = random.randrange(self.n_heads)

        # Training
        if self.reply_buffer.get_count() > self.min_buffer_size:
            self.get_session().run(self.train_op, feed_dict=self.generate_feed_dict())

        # Synchronize target network
        self.step_count += 1
        if self.step_count % self.target_sync_interval == 0:
            self.sync_target()

    def generate_feed_dict(self):
        def get_action_values(input_node, output_node, state):
            return self.get_session().run(output_node, feed_dict={input_node: [state]})[0]

        batch = self.reply_buffer.sample_batch(self.batch_size)
        feed_dict = {node: [] for node in self.nn_inputs + self.nn_outputs}

        for i in range(self.batch_size):
            bootstrap_mask = self.mask_generator(self.n_heads)

            state = batch["state"][i]
            next_state = batch["next_state"][i]
            action = batch["action"][i]
            reward = batch["reward"][i]
            done = batch["episode_done"][i]

            for head in range(self.n_heads):
                # TODO: empty list?
                # Mask out some heads
                if not bootstrap_mask[head]:
                    continue

                # Update action value
                target_action_values = get_action_values(self.target_nn_inputs[head],
                                                         self.target_nn_heads[head],
                                                         next_state)
                updated_action_value = get_action_values(self.nn_inputs[head],
                                                         self.nn_heads[head],
                                                         state)
                updated_action_value = list(updated_action_value)

                if done:
                    learning_target = reward
                else:
                    learning_target = reward + self.reward_decay*np.max(target_action_values)

                updated_action_value[action] += \
                    self.td_learning_rate*(learning_target - updated_action_value[action])

                # Add to feed dict
                feed_dict[self.nn_inputs[head]].append(state)
                feed_dict[self.nn_outputs[head]].append(updated_action_value)

        return feed_dict

    def sync_target(self):
        self.get_session().run(self.op_sync)


def test():
    def render():
        print env.render(mode='ansi')
        print "Reward:", reward
        print "Head:", agent.current_head
        print "Done:", done
        print ""
        time.sleep(frame_time)

    import time
    frame_time = 0.1

    env = snake.SnakeGame(3, 3, 1, 1, max_episode_length=20)
    agent = BootstrappedDQN(observation_space=env.observation_space,
                            action_space=env.action_space,
                            reward_decay=1.,
                            td_learning_rate=0.5,
                            target_sync_interval=200,
                            nn_constructor=nn_constructor,
                            loss_function=loss_function,
                            trainer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize,
                            replay_buffer_class=hrl.playback.MapPlayback,
                            replay_buffer_args={"capacity": 20000},
                            min_buffer_size=100,
                            batch_size=20)
    next_state = np.array(env.state)
    while True:
        state = next_state
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        render()

        agent.reinforce_(state=state,
                         action=action,
                         reward=reward,
                         next_state=next_state,
                         episode_done=done)

        if done:
            next_state = np.array(env.reset())
            render()

def loss_function(output, target):
    return tf.reduce_sum(tf.squared_difference(output, target))


def nn_constructor(observation_space, action_space, n_heads, **kwargs):
    # use different weights for each head
    def leakyRelu(x):
        return tf.maximum(0.01*x, x)

    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

    def weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    eshape = observation_space.shape
    nn_inputs = []
    nn_outputs = []

    n_channel1 = 8
    w1 = weight([3, 3, eshape[-1], n_channel1])
    b1 = bias([n_channel1])

    n_channel2 = 16
    w2 = weight([n_channel1*eshape[0]*eshape[1], n_channel2])
    b2 = bias([n_channel2])

    for i in range(n_heads):
        x = tf.placeholder(tf.float32, (None,) + observation_space.shape)

        w3 = weight([n_channel2, 4])
        b3 = bias([4])

        layer1 = leakyRelu(conv2d(x, w1) + b1)
        layer1_flatten = tf.reshape(layer1, [-1, n_channel1*eshape[0]*eshape[1]])

        layer2 = leakyRelu(tf.matmul(layer1_flatten, w2) + b2)

        layer3 = tf.matmul(layer2, w3) + b3

        nn_inputs.append(x)
        nn_outputs.append(layer3)

    return {"input": nn_inputs, "head": nn_outputs}


def test_bernoulli_mask():
    while True:
        print bernoulli_mask(0.5)(10)
        raw_input()

if __name__ == "__main__":
    test()
    # test_bernoulli_mask()
