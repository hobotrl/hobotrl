import os
import time
import tensorflow as tf


class BaseEnvironmentRunner(object):
    def __init__(self, env, agent,
                 n_episodes=-1, moving_average_window_size=50,
                 no_reward_reset_interval=-1,
                 checkpoint_save_interval=-1, log_dir=None, log_file_name=None,
                 render_env=False, render_interval=1, render_length=200, frame_time=0, render_options={}):
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

        if log_file_name:
            assert log_dir
            self.log_file = open(os.path.join(log_dir, log_file_name), "w")

        self.render_env = render_env
        self.render_interval = render_interval
        self.render_length = render_length
        self.frame_time = frame_time
        self.render_options = render_options

        self.episode_count = 0
        self.step_count = 0
        self.reward_history = [0]
        self.time_last_reward = 0

    def run(self):
        state = self.env.reset()
        while self.episode_count != self.n_episodes:
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

            if done:
                if self.log_file:
                    print "Episode %d Step %d:" % (self.episode_count, self.step_count),
                    print "%.2f/%.2f" % (self.reward_history[-2], self.reward_summary)

                    self.log_file.write("%f,%f\n" % (self.reward_history[-2], self.reward_summary))

            if self.checkpoint_save_interval != -1 and self.step_count % self.checkpoint_save_interval == 0:
                saver = tf.train.Saver()
                saver.save(self.agent.get_session(), os.path.join(self.log_dir, '%d.ckpt' % self.step_count))
                print "Checkpoint saved for at step %d" % self.step_count

            self.step_count += 1

    def step(self, state):
        action = self.agent.act(state)
        next_state, reward, done, info = self.env.step(action)

        self.reward_history[-1] += reward

        if reward > 1e-6:
            self.time_last_reward = self.step_count

        if self.step_count - self.time_last_reward == self.no_reward_reset_interval:
            print "Reset for no reward"
            done = True

        self.agent.reinforce_(state=state,
                              action=action,
                              reward=reward,
                              next_state=next_state,
                              episode_done=done)

        if done:
            next_state = self.env.reset()
            self.add_reward()
            self.time_last_reward = self.step_count
            self.episode_count += 1

        return next_state, done

    def add_reward(self):
        self.reward_history.append(0)

        if len(self.reward_history) > self.moving_average_window_size:
            del self.reward_history[0]

    @property
    def reward_summary(self):
        return float(sum(self.reward_history[:-1]))/(len(self.reward_history)-1)
