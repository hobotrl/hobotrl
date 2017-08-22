
import hobotrl as hrl


class LoadedAgent(hrl.tf_dependent.base.BaseDeepAgent):
    def __init__(self, checkpoint_path, save_path, input_name, output_name, sess=None, graph=None, global_step=None, **kwargs):
        super(LoadedAgent, self).__init__(sess, graph, global_step, **kwargs)
        self._preds = ...
        self._input_state = ...
        # todo

    def act(self, state, **kwargs):
        return self.sess.run([self._preds], feed_dict={self._input_state: np.asarray([state])})[0]

    def reinforce_(self, state, action, reward, next_state, episode_done=False, **kwargs):


class QueryExpertAgent(hrl.core.Agent):


    def __init__(self, agent, *args, **kwargs):
        super(QueryExpertAgent, self).__init__(*args, **kwargs)
        self._agent = agent
        self._thread = QueryThread(self._agent, self._queue, self._infoq)
        self._thread.start()


    def act(self, state, **kwargs):
        return self._agent.act(state, **kwargs)

    def step(self, state, action, reward, next_state, episode_done=False, **kwargs):
        return self._agent.step(...)


class QueryThread(threading.Thread):

    def __init__(self, agent, step_queue, info_queue, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        """

        :param agent:
        :type agent: Agent
        :param step_queue: queue filled by AsynchronousAgent with step()s for inner agent's step()
        :type step_queue: Queue.Queue
        :param info_queue: queue filled by this class from inner agent's step() returns
                for AsynchronousAgent to pass back as the return value of AsynchronousAgent.step()
        :type info_queue: Queue.Queue
        :param group:
        :param target:
        :param name:
        :param args:
        :param kwargs:
        :param verbose:
        """
        super(TrainingThread, self).__init__(group, target, name, args, kwargs, verbose)
        sample = self._replay.sample_batch()
        rospy.pub(sample, topic)
        action = subscribe(action_topic)
        state, action
        self._agent, self._step_queue, self._info_queue = agent, step_queue, info_queue
        self._stopped = False

    def run(self):
        while not self._stopped:
            step = self._step_queue.get(block=True)
            queue_empty = self._step_queue.qsize() == 0
            # async_buffer_end signal for asynchronous samplers, representing end of step queue
            info = self._agent.step(*step["args"], async_buffer_end=queue_empty, **step["kwargs"])
            self._info_queue.put(info)

    def stop(self):
        self._stopped = True
