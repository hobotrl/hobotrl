

"""
* weight sharing / op sharing / complex network structure
* single threaded / distributed
* policy based / value based
* lstm
"""


class Agent(object):
    def step(self, state, action, reward, next_state, episode_done):
        raise NotImplementedError()

    def act(self, state):
        raise NotImplementedError()


class Policy(object):
    def act(self, state):
        raise NotImplementedError()


class Function(object):
    def __call__(self, *args, **kwargs):
        pass


class NetworkFunction(Function):
    def __init__(self, symbol, inputs):
        """
        :param symbol:
        :type symbol: NetworkSymbol
        :param inputs: list of input symbol
        :type inputs: list
        """
        super(NetworkFunction, self).__init__()

    def __call__(self, *args):
        # function backed by network
        pass


class NetworkSymbol(object):
    def __init__(self, op, name, network):
        """
        :param op:
        :type op: tf.Tensor
        :param name:
        :param network:
        """
        super(NetworkSymbol, self).__init__()
        self.op, self._name, self._network = op, name, network

    # symbol in network

n1 = Network()

n2 = n1(inputs=[input_action, input_state])


class Network(object):

    def __init__(self, inputs, network_creator, var_scope, name_scope=None):
        if type(var_scope) == str:
            var_scope = tf.variable_scope(var_scope)
        self._var_scope, self._f_creator = var_scope, network_creator
        self._var_scope.__enter__()
        try:
            net = network_creator(inputs)
        finally:
            self._var_scope.__exit__()
        self._symbols = dict([(k, NetworkSymbol(net[k], k, self)) for k in net])

    def __getitem__(self, item):
        """
        return NetworkSymbol defined in this network, by network_creator
        :param item:
        :return:
        """
        return self._symbols[item]

    def __call__(self, inputs, name_scope="", *args, **kwargs):
        """
        :param inputs:
        :param args:
        :param kwargs:
        :return: another network created by this network's network_creator and possibly share weights
        """
        self._var_scope.reuse(True)
        return Network(inputs, self._f_creator, self._var_scope)

    def variables(self):
        # return trainable variables in this network
        pass


class NetworkWithTarget(Network):
    def __init__(self, network_creator, var_scope, target_var_scope):
        super(NetworkWithTarget, self).__init__(network_creator, var_scope)
        self._target = Network(network_creator, target_var_scope)
        self.init_syncer()

    def sync_target(self, rate, sess):
        sess.run([self.sym_sync], feed_dict={input_sync_rate: rate})

    @property
    def target(self):
        return self._target


class NetworkUpdater(object):

    def sym_loss(self, *args):
        pass

    def update(self, sess, *args, **kwargs):
        pass


class NetworkOptimizer(object):
    pass


class Sampler(object):
    def sample(self, replay):
        raise NotImplementedError()


class TrajectorySampler(Sampler):
    pass


class TransitionSampler(Sampler):
    pass


class TruncatedTrajectorySampler(Sampler):
    pass


class PERSampler(Sampler):
    pass


class EpsilonGreedyPolicy(Policy):

    def __init__(self, q_function):
        super(EpsilonGreedyPolicy, self).__init__()
        self.q_function = q_function

    def act(self, state):
        q_values = self.q_function(state)
        action = np.argmax(q_values)
        return action


class ValueBasedAgent(Agent):

    def __init__(self):
        super(ValueBasedAgent, self).__init__()
        self.init_value_function()
        self.init_policy()

    def init_value_function(self, **kwargs):
        self.q_function = None
        pass

    def init_policy(self):
        self.policy = EpsilonGreedyPolicy(self.q_function)

    def act(self, state):
        return self.policy.act(state)


class ReplayMemoryAgent(Agent):

    def __init__(self):
        super(ReplayMemoryAgent, self).__init__()
        self.init_memory()

    def init_memory(self):
        self.replay = MapPlayback()

    def step(self, state, action, reward, next_state, episode_done):
        self.replay.push_sample(self.make_sample(state, action, reward, next_state, episode_done))
        self.update_on_memory()
        pass

    def make_sample(self, state, action, reward, next_state, episode_done):
        pass

    def update_on_memory(self):
        raise NotImplementedError()


class TrajectoryBatchUpdate(ReplayMemoryAgent):

    def update_on_memory(self):
        return self.update_on_trajectory(self.sample_trajectory(batch_size))

    def sample_trajectory(self, batch_size):
        # sample {batch_size} episodes from batch
        pass

    def update_on_trajectory(self, batch):
        raise NotImplementedError()


class TransitionBatchUpdate(TrajectoryBatchUpdate):

    def update_on_memory(self):
        return self.update_on_transition(self.sample_transition(batch_size))
        pass

    def sample_transition(self, batch_size):
        return self.replay.sample_batch(batch_size)
        # can be override by sub class
        pass

    def update_on_trajectory(self, batch):
        pass

    def update_on_transition(self, batch):
        raise NotImplementedError()


class OneStepTD(NetworkUpdater):

    def __init__(self, learn_q, target_q):
        super(OneStepTD, self).__init__()
        self.input_target_q = tf.placeholder()
        self.input_action = tf.placeholder()
        self._sym_loss = Network.clipped_square(self.input_target_q - tf.reduce_sum(tf.one_hot(self.input_action) * learn_q.op))

    def sym_loss(self, *args):
        return self._sym_loss

    def update(self, sess, batch, *args, **kwargs):
        if not ddqn:
            target_q_val = self.target_q(batch["next_state"])
            target_q_val = np.max(target_q_val, axis=1)
        else:
            learn_q_val = self.learn_q(batch["next_state"])
            target_action = np.argmax(learn_q_val, axis=1)
            target_q_val = np.sum(self.target_q(batch["next_state"]) * np.one_hot(target_action), axis=1)
        return MinimizeLoss([self._sym_loss], self.learn_q.input_dict([batch["state"]])+{self.input_target_q: target_q_val})


class DQN(TransitionBatchUpdate, ValueBasedAgent):

    def __init__(self, network_optimizer, **kwargs):
        super(DQN, self).__init__()
        self.network_optimizer = network_optimizer
        network_optimizer.add_updater(OneStepTD(self.learn_q, self.target_q), name="td")
        network_optimizer.add_updater(L2(self.learnable_network), weight=1e-4, name="l2")

    def init_value_function(self, **kwargs):
        super(DQN, self).init_value_function(**kwargs)
        self.network = NetworkWithTarget(f_creator, var_scope="learn", target_var_scope="target")
        learn_q = NetworkFunction(self.network["q"], input=[input_state])
        target_q = NetworkFunction(self.network.target["q"], input=[input_state])
        self.q_function = learn_q

    def update_on_transition(self, batch):
        self.network_optimizer.updater("td").update(self.sess, batch)
        self.network_optimizer.updater("l2").update(self.sess)

        self.network_optimizer.optimize_step()
        if a % step == 0:
            self.network.sync_target()


class ActorCritic(Agent):
    def __init__(self, network_optimizer, **kwargs):
        super(ActorCritic, self).__init__()
        self.init_actor_critic()
        self.init_policy()
        self.on_sampler = TrajectorySampler(max_len=16, size=1, on=True)
        self.off_sampler = TransitionSampler(batch_size=32, interval=32, memory=MapPlayback())
        network_optimizer.add_updater(NStepAC(self.action_value, self.dist_func), name="ac")
        network_optimizer.add_updater(OneStepTD(self.action_value, self.action_value), name="off_td")

    def init_policy(self):
        self.policy = DistributionPolicy(self.policy_distribution)

    def act(self, state):
        return self.policy.act(state)

    def init_actor_critic(self, is_continuous):
        self.network = Network(f_creator, var_scope="ac")
        self.action_value = NetworkFunction(self.network["q"], [input_state])
        self.dist_func = NetworkFunction(self.network["pi"], [input_state])
        self.policy_distribution = DiscreteDistribution(self.dist_func)

    def step(self, state, action, reward, next_state, episode_done):
        has_update = False
        batch = self.on_sampler.step(state, action, reward, next_state, episode_done)
        if batch is not None:
            has_update = True
            self.network_optimizer.updater("ac").update(self.sess, batch)
        batch = self.off_sampler.step(state, action, reward, next_state, episode_done)
        if batch is not None:
            has_update = True
            self.network_optimizer.updater("off_td").update(self.sess, batch)
        if has_update:
            self.network_optimizer.optimize_step()


class ActorCriticAgent(TrajectoryBatchUpdate, ActorCritic):
    pass


