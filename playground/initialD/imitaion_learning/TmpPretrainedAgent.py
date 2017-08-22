import tensorflow as tf
import numpy as np
from hobotrl.tf_dependent.base import BaseDeepAgent
import os


class TmpPretrainedAgent(BaseDeepAgent):
    def __init__(self, checkpoint_path, train_dir, input_name, output_name, trainop_name,
                 sess=None, graph=None, global_step=None, **kwargs):


        self.checkpoint_path = checkpoint_path
        self.train_dir = train_dir
        self.input_name = input_name
        self.output_name = output_name
        self.trainop_name = trainop_name
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        super(TmpPretrainedAgent, self).__init__(sess=sess, graph=graph, global_step=global_step, **kwargs)



    def init_network(self, *args, **kwargs):
        checkpoint_path = self.checkpoint_path
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta", clear_devices=True)
        saver.restore(self.sess, checkpoint_path)
        # Not sure !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.graph = tf.get_default_graph()
        self.inputs = self.graph.get_tensor_by_name(self.input_name)
        self.outputs = self.graph.get_tensor_by_name(self.output_name)
        self.trainop = self.graph.get_operator_by_name(self.trainop_name)
        # This interfave is very strange and needs removed

        self.is_train = self.graph.get_tensor_by_name('is_train').outputs[0]

    def act(self, state, **kwargs):
        action = self.sess.run(self.outputs, feed_dict={
            self.inputs: np.repeat(
                state[np.newaxis, :, :, :],
                axis=0,
                repeats=256),
            self.is_train: False,
        })[0]

        return action

    def learn(self, replay_buffer, update_n):
        replay_size = len(replay_buffer)
        num_iter = 100
        batch_size = 100
        for i in range(num_iter):
            batch = replay_buffer[np.random.randint(replay_size, size=batch_size)]
            self.sess.run(self.trainop, feed_dict={self.inputs: batch})
        save_path = os.path.join(self.train_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=(update_n+1)*num_iter)


class QueryAgent(TmpPretrainedAgent):
    def __init__(self, sess=None, graph=None, global_step=None, checkpoint_path=None, input_name=None, output_name=None, **kwargs):

        super(QueryAgent, self).__init__(checkpoint_path, input_name, output_name, sess, graph, global_step, **kwargs)

    def act(self, state, **kwargs):
        pass



