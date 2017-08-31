import tensorflow as tf
import numpy as np
from hobotrl.tf_dependent.base import BaseDeepAgent

class PretrainedAgent(BaseDeepAgent):
    def __init__(self, checkpoint_path, input_name, output_name,
                 sess=None, graph=None, global_step=None, **kwargs):
        super(PretrainedAgent, self).__init__(sess, graph, global_step, **kwargs)
        saver = tf.train.import_meta_graph(checkpoint_path+".meta", clear_devices=True)
        saver.restore(self.sess, checkpoint_path)
        self.inputs = self.graph.get_tensor_by_name(input_name)
        self.outputs = self.graph.get_tensor_by_name(output_name)
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





