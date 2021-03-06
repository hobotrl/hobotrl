import hobotrl as hrl
import tensorflow as tf
import numpy as np
import random
import time


def bernoulli_mask(p):
    """
    Return a function that can generate bootstrap mask using bernoulli distribution.

    :param p: parameter for bernoulli distribution
    :return: a mask generator that take an integer 'n' as input and generates 'n' masks.
    """
    return lambda n: [np.random.binomial(1, p) for _ in range(n)]


class BootstrappedDQN(hrl.tf_dependent.base.BaseDeepAgent):
    def __init__(self, observation_space, action_space,
                 nn_constructor, loss_function, trainer,
                 reward_decay, td_learning_rate, target_sync_interval,
                 replay_buffer_class, replay_buffer_args, min_buffer_size, batch_size=20,
                 n_heads=10, bootstrap_mask=bernoulli_mask(0.5)):
        """
        A bootstrapped DQN.
        Based on arXiv:1602.04621 [cs.LG]

        :param observation_space: the environment's observation space.
        :param action_space: the environment's action space.
        :param nn_constructor(callable): a constructor that constructs the neurual network.
            param observation_space: the environment's observation space.
            param action_space: the environment's action space.
            return: a dictionary.
                The key "input" contains a list of input nodes(placeholder) for each head;
                the key "head" contains a list of output nodes for each head.
        :param loss_function(callable): represents the loss function.
            param output: node for the neural network's output .
            param target: node for the training target.
            return: a tensorflow node that represents a list of training loss for each sample.
        :param trainer(callable): a trainer.
            param loss: the training loss.
            return: a tensorflow operation that can train the neural network.
        :param reward_decay(float): reward decay(lambda).
        :param td_learning_rate(float): learning rate for temporal difference learning(alpha).
        :param target_sync_interval(int): controls the frequency of synchronizing the target network.
        :param replay_buffer_class(class): a class that can be used as a replay buffer.
            param sample_shapes: shape for each sample.
        :param replay_buffer_args(dict): arguments that will be passed to "replay_buffer_class.__init__()".
        :param min_buffer_size(int): start training after the buffer grows to this size.
        :param batch_size(int): size for each batch of training data.
        :param n_heads(int): number of heads.
        :param bootstrap_mask(callable): a bootstrap_mask generator.
        """
        assert 0. <= reward_decay <= 1.
        assert 0 < td_learning_rate <= 1.
        assert target_sync_interval > 0
        assert callable(nn_constructor)
        assert callable(loss_function)
        assert callable(trainer)
        assert issubclass(replay_buffer_class, hrl.playback.MapPlayback)
        assert n_heads > 0
        assert callable(bootstrap_mask)

        # Create tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        super(BootstrappedDQN, self).__init__(sess=session)

        # Initialize parameters
        self.observation_space = observation_space

        self.action_space = action_space
        try:
            self.action_space_shape = (action_space.n,)
        except AttributeError:
            self.action_space_shape = action_space.shape

        self.nn_constructor = nn_constructor
        self.reward_decay = reward_decay
        self.td_learning_rate = td_learning_rate
        self.target_sync_interval = target_sync_interval
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.mask_generator = bootstrap_mask

        self.masks = [tf.placeholder(tf.float32, shape=(None,)) for head in range(self.n_heads)] \
            # Placeholder for bootstrap masks
        self.current_head = 0  # The head for current episode
        self.step_count = 0  # Total number of steps for all episodes

        # Initialize the replay buffer
        self.reply_buffer = replay_buffer_class(sample_shapes={
                                                    "state": observation_space.shape,
                                                    "mask": (self.n_heads,),
                                                    "next_state": observation_space.shape,
                                                    "action": [],
                                                    "reward": [],
                                                    "episode_done": [],
                                                },
                                                **replay_buffer_args)

        # Construct the neural network
        # Non-target network
        with tf.variable_scope('non-target') as scope_non_target:
            nn = nn_constructor(observation_space=observation_space,
                                action_space=action_space,
                                n_heads=n_heads)
            assert len(nn["head"]) == self.n_heads

            self.nn_input = nn["input"]
            self.nn_heads = nn["head"]
            self.nn = nn

        # Target network
        with tf.variable_scope('target') as scope_target:
            nn = nn_constructor(observation_space=observation_space,
                                action_space=action_space,
                                n_heads=n_heads)
            assert len(nn["head"]) == self.n_heads

            self.target_nn_input = nn["input"]
            self.target_nn_heads = nn["head"]

        # Construct synchronize operation
        non_target_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=scope_non_target.name)
        target_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=scope_target.name)
        self.op_sync_target = [tf.assign(target_var, non_target_var)
                               for target_var, non_target_var
                               in zip(target_vars, non_target_vars)]
        self.non_target_vars = non_target_vars

        # Construct training operation
        self.nn_outputs = [tf.placeholder(tf.float32, (None, action_space.n))
                           for _ in range(self.n_heads)]
        nn_output = tf.concat(self.nn_heads, 0)
        nn_target = tf.concat(self.nn_outputs, 0)
        masks = tf.concat(self.masks, 0)

        loss_list = loss_function(output=nn_output, target=nn_target)
        self.loss = tf.reduce_sum(tf.multiply(loss_list, masks))  # Apply bootstrap mask
        self.op_train = trainer(self.loss)

        # Initialize the neural network
        self.get_session().run(tf.global_variables_initializer())
        self.get_session().run(self.op_sync_target)

    def act(self, state, show_action_values=False, **kwargs):
        """
        Choose an action to take.

        :param state: current state.
        :param show_action_values: whether to print action values
        :return: an action.
        """
        action_values = self.get_session().run(self.nn_heads[self.current_head],
                                               {self.nn_input: [state]})[0]

        # Print action values if needed
        if show_action_values:
            print action_values

        return np.argmax(action_values)

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=None, **kwargs):
        """
        Saves training data and train the neural network.
        Asserts that "state" and "next_state" are already numpy arrays.

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param episode_done:
        :return: loss.
        """
        # TODO: don't give the default value for "episode_done" in the base class
        assert episode_done is not None

        info = {}

        # Add to buffer
        self.reply_buffer.push_sample({"state": state,
                                       "next_state": next_state,
                                       "action": np.asarray(action),
                                       "reward": np.asarray(reward),
                                       "episode_done": np.asarray(episode_done),
                                       "mask": np.array(self.mask_generator(self.n_heads))})

        # Randomly choose next head if this episode ends
        if episode_done:
            self.current_head = random.randrange(self.n_heads)

        # Training
        if self.step_count > self.min_buffer_size:
            batch = self.reply_buffer.sample_batch(self.batch_size)
            feed_dict = self.generate_feed_dict(batch)
            loss = self.train(feed_dict)
        else:
            loss = float("nan")

        # Synchronize target network
        self.step_count += 1

        if self.step_count % self.target_sync_interval == 0:
            self.sync_target()

        info["loss"] = loss
        return info

    def train(self, feed_dict):
        """
        Train the neural network with data in feed_dict.

        :param feed_dict: data to be passed to tensorflow.
        :return: loss.
        """
        try:
            _, loss = self.get_session().run([self.op_train, self.loss], feed_dict=feed_dict)
            return loss
        except ValueError:  # If all masks happens to be zero, i.e. there's no sample
            return 0.0

    def generate_feed_dict(self, batch):
        """
        Generate "feed_dict" for tf.Session().run() using next batch's data.

        :return(dict): "feed_dict"
        """
        def get_action_values(input_node, output_node, state):
            """
            Calculate action values.

            :param input_node: neural network's input node.
            :param output_node: neural network's output node.
            :param state: game state.
            :return(numpy.ndarray): action values.
            """
            return self.get_session().run(output_node, feed_dict={input_node: state})

        feed_dict = {node: [] for node in [self.nn_input] + self.nn_outputs + self.masks}

        next_state_action_values = get_action_values(self.target_nn_input, self.target_nn_heads, batch["next_state"])
        current_state_action_values = get_action_values(self.nn_input, self.nn_heads, batch["state"])

        for i in range(self.batch_size):
            # Unpack data
            state = batch["state"][i]
            action = batch["action"][i]
            reward = batch["reward"][i]
            done = batch["episode_done"][i]
            bootstrap_mask = batch["mask"][i]

            # Add current state to training data
            feed_dict[self.nn_input].append(state)

            for head in range(self.n_heads):
                # Get old action values for current head
                target_action_values = next_state_action_values[head][i]
                updated_action_values = list(current_state_action_values[head][i])

                # Add bootstrap mask to training data
                feed_dict[self.masks[head]].append(bootstrap_mask[head])

                # Mask out some heads
                if not bootstrap_mask[head]:
                    feed_dict[self.nn_outputs[head]].append(updated_action_values)
                    continue

                # Calculate new action values
                if done:
                    learning_target = reward
                else:
                    learning_target = reward + self.reward_decay * np.max(target_action_values)

                updated_action_values[action] += \
                    self.td_learning_rate * (learning_target - updated_action_values[action])

                # Add new action values to training data
                feed_dict[self.nn_outputs[head]].append(updated_action_values)

        return feed_dict

    def sync_target(self):
        """
        Update the target network.
        """
        self.get_session().run(self.op_sync_target)


class RandomizedBootstrappedDQN(BootstrappedDQN):
    def __init__(self, eps_function, **args):
        """
        :param eps_function(callable): maps step count to epsilon.
        :param args: other arguments that will be passed to "BootstrappedDQN".
        """
        super(RandomizedBootstrappedDQN, self).__init__(**args)
        self.eps_function = eps_function

    def act(self, state, **kwargs):
        if random.random() < self.eps_function(self.step_count):
            return self.action_space.sample()
        else:
            return super(RandomizedBootstrappedDQN, self).act(state, **kwargs)


class CEMBootstrappedDQN(BootstrappedDQN):
    def __init__(self, cem_update_interval, cem_portion, cem_noise, cem_max_variance=5, **kwargs):
        """
        Bootstrapped DQN combined with cross-entropy method.

        :param cem_update_interval: update parameters every this number of episodes.
        :param cem_portion: pass to CrossEntropyMethodParameterGenerator.
        :param cem_noise: pass to CrossEntropyMethodParameterGenerator.
        :param cem_max_variance: pass to CrossEntropyMethodParameterGenerator.
        :param kwargs: pass to super.
        """
        from hobotrl.algorithms.cross_entropy_method import CrossEntropyMethodParameterGenerator
        super(CEMBootstrappedDQN, self).__init__(**kwargs)

        self.nn_head_para = self.nn["head_para"]  # parameter list for each head
        self.cem_update_interval = cem_update_interval

        self.episode_count = 0  # Episode counter
        self.reward_records = [list() for _ in range(self.n_heads)]  # Record reward for each head
        self.current_reward_record = 0.

        # Prepare for cross-entropy method
        para_shapes = [para.shape for para in self.nn_head_para[0]]
        self.cem = CrossEntropyMethodParameterGenerator(parameter_shapes=para_shapes,
                                                        n=self.n_heads,
                                                        proportion=cem_portion,
                                                        initial_variance=0,
                                                        noise=cem_noise,
                                                        max_variance=cem_max_variance)

    def reinforce_(self, state, action, reward, next_state,
                   episode_done=None, **kwargs):

        # Count rewards
        self.current_reward_record += reward

        if episode_done:
            # Count episode
            self.episode_count += 1

            # Update Reward Record
            self.reward_records[self.current_head].append(self.current_reward_record)
            self.current_reward_record = 0.

            # Update parameters
            if self.episode_count % self.cem_update_interval == 0:
                self.update_parameters()
                self.reward_records = [list() for _ in range(self.n_heads)]

        return super(CEMBootstrappedDQN, self).reinforce_(state, action, reward, next_state,
                                                          episode_done, **kwargs)

    def update_parameters(self):
        print "CEM update"

        # Retrieve parameters and summarize scores
        para_lists = [self.get_session().run(para_list) for para_list in self.nn_head_para]
        scores = [np.mean(reward_list) if len(reward_list) > 0
                  else float("-inf")
                  for reward_list in self.reward_records]

        # Update parameters
        self.cem.update_parameter_lists(parameter_lists=para_lists, scores=scores)

        # Assign parameters
        op_update = []
        for (list_id, para_list) in enumerate(para_lists):
            for (para_id, para) in enumerate(para_list):
                op_update.append(self.nn_head_para[list_id][para_id].assign(para))
        self.get_session().run(op_update)
