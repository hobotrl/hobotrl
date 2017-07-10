import sys
sys.path.append('../../../')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.tf_dependent.policy import DeepStochasticPolicy

def f_net_discrete(inputs, num_actions):
    depth_state = inputs[0].get_shape()[1:].num_elements()
    inputs = tf.reshape(inputs, shape=[-1, depth_state], name='inputs')
    action_dist = layers.dense(
        inputs=inputs, units=num_actions,
        activation=tf.nn.softmax,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True
    )
    return action_dist

optimizer_spg = tf.train.GradientDescentOptimizer(learning_rate=0.001)
training_params = (optimizer_spg,)
state_shape = (99, 99, 3)
num_actions = 4
batch_size = 10
graph = tf.get_default_graph()

print "=============="
print "Test initialize Policy : ",
dsp = DeepStochasticPolicy(
    state_shape=state_shape, num_actions=num_actions,
    is_continuous_action=False, f_create_net=f_net_discrete,
    training_params=training_params, entropy=0.01
)
print 'pass!\n'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

args_s = [batch_size] + list(state_shape)
state = np.random.rand(*args_s)
action = np.random.randint(0, num_actions, [batch_size])
advantage = np.random.rand(*[batch_size])
feed_dict = {
    dsp.input_state: state,
    dsp.input_action: action,
    dsp.input_advantage: advantage
}

print "================="
print "Test improve policy:"
print "Initial dpg_loss: ",
print sess.run(dsp.spg_loss, feed_dict)
for i in range(20):
    dsp.improve_policy_(
        state=state, action=action,
        advantage=advantage, sess=sess
    )
    print "dpg_loss after pg step {}".format(i),
    print sess.run(dsp.spg_loss, feed_dict)
print 'pass!\n'

print "================="
print "Test act:"
print "Case 1: batch:"
print dsp.act(state=state, sess=sess).shape
print "pass!\n"

print "Case 2: single sample (should raise exception):"
try:
    print dsp.act(state[0, :], sess=sess).shape
except Exception, error:
    print str(error)
finally:
    print "pass!\n"

print "Case 3: single sample with batch dim.:"
print dsp.act(state[0, :][np.newaxis, :], sess=sess).shape
print "pass!\n"

sess.close()


