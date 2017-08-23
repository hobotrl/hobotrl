import tensorflow as tf
import numpy as np
from hobotrl.tf_dependent.base import BaseDeepAgent
import os
import sys
sys.path.append("/home/pirate03/anaconda2/lib/python2.7/site-packages")
import sklearn.metrics



class TmpPretrainedAgent(BaseDeepAgent):
    def __init__(self, checkpoint_path, train_dir, input_name, output_name, trainop_name,
                 sess=None, graph=None, global_step=None, **kwargs):


        self.checkpoint_path = checkpoint_path
        self.train_dir = train_dir
        self.input_name = input_name
        self.output_name = output_name
        self.trainop_name = trainop_name
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        self.update_n = 0
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

    def learn(self, replay_buffer):
        if self.update_n == 0:
            preds = self.sess.run(self.outputs, feed_dict={self.inputs: replay_buffer, self.is_train: False})
            y_true = np.array([y[1] for y in replay_buffer])
            print "update_n: {}".format(self.update_n)
            self.evaluate(y_true, preds)
        self.evaluate(y_true, preds)

        replay_size = len(replay_buffer)
        batch_size = 128
        num_update = replay_size * 10 / batch_size
        for i in range(num_update):
            batch = replay_buffer[np.random.randint(replay_size, size=batch_size)]
            self.sess.run(self.trainop, feed_dict={self.inputs: batch, self.is_train: True})
        self.update_n += 1
        preds = self.sess.run(self.outputs, feed_dict={self.inputs: replay_buffer, self.is_train: False})
        y_true = np.array([y[1] for y in replay_buffer])
        print "update_n: {}".format(self.update_n)
        self.evaluate(y_true, preds)
        save_path = os.path.join(self.train_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=(self.update_n+1)*num_update)

    def evaluate(self, y_true, preds):
        prec = sklearn.metrics.precision_score(y_true, preds, average=None)
        rec = sklearn.metrics.recall_score(y_true, preds, average=None)
        f1 = sklearn.metrics.f1_score(y_true, preds, average=None)
        conf_mat = sklearn.metrics.confusion_matrix(y_true, preds)
        print "val_prec: {}".format(prec)
        print "val_rec: {}".format(rec)
        print "val_f1: {}".format(f1)
        print "val_conf_mat: {}".format(conf_mat)


class QueryAgent(TmpPretrainedAgent):
    def __init__(self, sess=None, graph=None, global_step=None, checkpoint_path=None, input_name=None, output_name=None, **kwargs):

        super(QueryAgent, self).__init__(checkpoint_path, input_name, output_name, sess, graph, global_step, **kwargs)

    def act(self, state, **kwargs):
        pass



