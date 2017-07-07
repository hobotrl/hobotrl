import os
import time
import tensorflow as tf


class BaseEnvironmentRunner(object):
    def __init__(self, env, agent,
                 n_episodes=-1, moving_average_window_size=50,
                 no_reward_reset_interval=-1,
                 checkpoint_save_interval=-1, log_dir=None, log_file_name=None,
                 render_env=False, render_interval=1, render_length=200, frame_time=0, render_options={},
                 show_frame_rate=False, show_frame_rate_interval=100):
        """
        Trains the agent in the environment.

        :param env: the environment.
        :param agent: the agent.
        :param n_episodes: number of episodes before terminating, -1 means run forever.
        :param moving_average_window_size: window size for calculating moving average of rewards.
        :param no_reward_reset_interval: reset after this number of steps if no reward is received.
        :param checkpoint_save_interval: save checkpoint every this number of steps.
        :param log_dir: path to save log files.
        :param log_file_name: file name of the csv file.
        :param render_env: whether to render the environment during the training.
        :param render_interval: number of steps between each render session.
        :param render_length: length of render session, counted in steps.
        :param frame_time: time interval between each frame, in seconds.
        :param render_options: options that will pass to env.render().
        """
        assert n_episodes >= -1
        assert moving_average_window_size >= 1
        assert no_reward_reset_interval >= -1
        assert checkpoint_save_interval >= -1
        assert render_interval >=1
        assert render_length > 0
        assert frame_time >= 0

        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.moving_average_window_size = moving_average_window_size
        self.no_reward_reset_interval = no_reward_reset_interval
        self.checkpoint_save_interval = checkpoint_save_interval
        self.log_dir = log_dir
        self.render_env = render_env
        self.render_interval = render_interval
        self.render_length = render_length
        self.frame_time = frame_time
        self.render_options = render_options
        self.show_frame_rate = show_frame_rate
        self.show_frame_rate_interval = show_frame_rate_interval

        # Open log file
        if log_file_name:
            assert log_dir
            self.log_file = open(os.path.join(log_dir, log_file_name), "w")
        else:
            self.log_file = None

        # Open summary writer
        if log_dir:
            self.summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
        else:
            self.summary_writer = None

        self.episode_count = 0  # Count episodes
        self.step_count = 0  # Count number of total steps
        self.reward_history = [0]  # Record the total reward of last episodes
        self.time_last_reward = 0  # The time when the agent get last reward

    def run(self):
        """
        Start training.
        """
        # Initialize environment
        state = self.env.reset()

        last_frame_rate_checkpoint = time.time()

        while self.episode_count != self.n_episodes:
            # Run a step
            state, done = self.step(state)

            # Render
            if self.render_env and self.step_count % self.render_interval <= self.render_length:
                render_result = self.env.render(**self.render_options)

                if render_result:
                    print render_result

                time.sleep(self.frame_time)

                # Close the window
                if self.step_count % self.render_interval == self.render_length:
                    self.env.render(close=True)

            # Save data to log file
            if done:
                if self.log_file:
                    print "Episode %d Step %d:" % (self.episode_count, self.step_count),
                    print "%7.2f/%.2f" % (self.reward_history[-2], self.reward_summary)

                    self.log_file.write("%d,%d,%f,%f\n" % (self.episode_count, self.step_count, self.reward_history[-2], self.reward_summary))

                if self.summary_writer:
                    summary = tf.Summary()
                    summary.value.add(tag="step count", simple_value=self.step_count)
                    summary.value.add(tag="reward", simple_value=self.reward_history[-2])
                    summary.value.add(tag="average reward", simple_value=self.reward_summary)

                    self.summary_writer.add_summary(summary, self.episode_count)

            # Save checkpoint
            if self.checkpoint_save_interval != -1 and self.step_count % self.checkpoint_save_interval == 0:
                saver = tf.train.Saver()
                saver.save(self.agent.get_session(), os.path.join(self.log_dir, '%d.ckpt' % self.step_count))
                print "Checkpoint saved at step %d" % self.step_count

            self.step_count += 1

            # Calculate frame rate
            if self.show_frame_rate and self.step_count % self.show_frame_rate_interval == 0:
                print "Frame rate:", self.show_frame_rate_interval/(time.time() - last_frame_rate_checkpoint)
                last_frame_rate_checkpoint = time.time()

    def step(self, state):
        """
        Take a step.

        :param state: current state
        :return: a tuple: (next state, whether current episode is done)
        """
        # def calculate_time(prompt = ""):
        #     global st
        #     et = time.time()
        #
        #     if prompt and self.step_count > 201:
        #         try:
        #             print prompt, et - st
        #         except NameError:
        #             pass
        #
        #     st = time.time()

        # calculate_time("\nOther time")

        action = self.agent.act(state)

        # calculate_time("Action time")

        next_state, reward, done, info = self.env.step(action)
        if self.render_env:
            print reward

        # calculate_time("Step time")

        self.reward_history[-1] += reward

        # Reset if no reward is seen for last a few steps
        if reward > 1e-6:
            self.time_last_reward = self.step_count

        if self.step_count - self.time_last_reward == self.no_reward_reset_interval:
            print "Reset for no reward"
            done = True

        # calculate_time("Calculate time")

        # Train the agent
        self.agent.reinforce_(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              episode_done=done)

        # calculate_time("Reinforce time")

        # Episode done
        if done:
            next_state = self.env.reset()
            self.add_reward()
            self.time_last_reward = self.step_count
            self.episode_count += 1

        return next_state, done

    def add_reward(self):
        """
        Add a new record.
        """
        self.reward_history.append(0)

        if len(self.reward_history) > self.moving_average_window_size:
            del self.reward_history[0]

    @property
    def reward_summary(self):
        """
        Get the average reward of last episodes.
        """
        return float(sum(self.reward_history[:-1]))/(len(self.reward_history)-1)

    def run_demo(self, file_name):
        """
        Load a checkpoint and run a demo.

        :param file_name: the checkpoint's file name.
        """
        saver = tf.train.Saver()
        saver.restore(self.agent.get_session(), os.path.join(self.log_dir, file_name))

        state = self.env.reset()
        while True:
            action = self.agent.act(state)
            state, reward, done, info = self.env.step(action)

            render_result = self.env.render(**self.render_options)
            if render_result:
                print render_result

            if done:
                self.env.reset()

            time.sleep(self.frame_time)
