import sys
sys.path.append('../../../')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.tf_dependent.policy import DeepDeterministicPolicy

def f_net(inputs, action_shape, is_training):
    depth_state = inputs.get_shape()[1:].num_elements()
    inputs = tf.reshape(inputs, shape=[-1, depth_state], name='inputs')
    depth_action = reduce(lambda x, y: x*y, action_shape, 1)

    action = layers.dense(
        inputs=inputs, units=depth_action,
        activation=None,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True
    )
    action = tf.reshape(action, shape=[-1]+list(action_shape), name='out')
    
    return action

optimizer_dpg = tf.train.GradientDescentOptimizer(learning_rate=0.001)
training_params = (optimizer_dpg,)
state_shape = (99, 99, 3)
action_shape = (3, 2)
batch_size = 10
graph = tf.get_default_graph()

print "=============="
print "Test initialize QFunc : ",
ddp = DeepDeterministicPolicy(
    f_net=f_net, state_shape=state_shape, action_shape=action_shape,
    training_params=training_params, schedule=(3,),
    batch_size=batch_size,
    graph=None
)
print 'pass!\n'

print "================="
print "Non-target subgraph in and out:"
print ddp.get_subgraph_policy()
print 'pass!\n'

args_s = [batch_size] + list(state_shape)
args_a = [batch_size] + list(action_shape)
state = np.random.rand(*args_s)
grad_q_action = np.random.rand(*args_a)

feed_dict = {
    ddp.sym_state: state,
    ddp.sym_grad_q_action: grad_q_action
}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print "================="
print "Test apply_op_train_dpg_():"
print "Initial dpg_loss: ",
print sess.run(ddp.sym_dpg_loss, feed_dict)
for i in range(10):
    ddp.apply_op_train_dpg_(
        state=state, grad_q_action=grad_q_action,
        sess=sess
    )
    print "td_loss after td step {}".format(i),
    print sess.run(ddp.sym_dpg_loss, feed_dict)

print "================="
print "Test improve policy:"
print "Initial dpg_loss: ",
print sess.run(ddp.sym_dpg_loss, feed_dict)
for i in range(20):
    ddp.improve_policy_(
        state=state, grad_q_action=grad_q_action,
        sess=sess
    )
    print "counters dpg {}".format(ddp.countdown_dpg_),
    print "dpg_loss after td step {}".format(i),
    print sess.run(ddp.sym_dpg_loss, feed_dict)
print 'pass!\n'

print "================="
print "Test act:"
print "Case 1: batch:"
print ddp.act(state=state, sess=sess).shape
print "pass!\n"

print "Case 2: single sample (should raise exception):"
try:
    print ddp.act(state[0, :], sess=sess).shape
except Exception, error:
    print str(error)
finally:
    print "pass!\n"

print "Case 3: single sample with batch dim.:"
print ddp.act(state[0, :][np.newaxis, :], sess=sess).shape
print "pass!\n"

sess.close()


