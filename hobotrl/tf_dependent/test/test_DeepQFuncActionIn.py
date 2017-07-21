import sys
sys.path.append('../../../')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.tf_dependent.value_function import DeepQFuncActionIn

def f_net(inputs_state, inputs_action, is_training):
    depth = inputs_state.get_shape()[1:].num_elements()
    inputs_state = tf.reshape(inputs_state, shape=[-1, depth])
    depth = inputs_action.get_shape()[1:].num_elements()
    inputs_action = tf.reshape(inputs_action, shape=[-1, depth])
    inputs = tf.concat([inputs_state, inputs_action], axis=1, name='inputs')
    hidden1 = layers.dense(
        inputs=inputs, units=200,
        activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='hidden1',
    )
    hidden2 = layers.dense(
        inputs=hidden1, units=200,
        activation=tf.nn.relu,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='hidden2',
    )
    q = layers.dense(
        inputs=hidden2, units=1,
        activation=tf.nn.tanh,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='out',
    )
    q = tf.squeeze(q, axis=1, name='out_sqz')
    return q

optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
target_sync_rate = 0.01
training_params = (optimizer_td, target_sync_rate, 10.0)
state_shape = (99, 99, 3)
action_shape = (3, 2)
batch_size = 10
graph = tf.get_default_graph()

print "=============="
print "Test initialize QFunc : ",
dqn = DeepQFuncActionIn(
    gamma=0.99,
    f_net=f_net, state_shape=state_shape, action_shape=action_shape,
    training_params=training_params, schedule=(2,5),
    graph=None
)
print 'pass!\n'

print "================="
print "Non-target subgraph in and out:"
print dqn.get_subgraph_value()
print 'pass!\n'

print "================="
print "Target subgraph in and out:"
print dqn.get_subgraph_value_target()
print 'pass!\n'

args_s = [batch_size] + list(state_shape)
args_a = [batch_size] + list(action_shape)
state = np.random.rand(*args_s)
action = np.random.rand(*args_a)
reward = np.random.rand(batch_size)
next_state = np.random.rand(*args_s)
next_action = np.random.rand(*args_a)
importance = np.ones([batch_size])
episode_done = np.zeros([batch_size])

feed_dict = {
    dqn.sym_state: state,
    dqn.sym_action: action,
    dqn.sym_reward: reward,
    dqn.sym_next_state: next_state,
    dqn.sym_next_action: next_action,
    dqn.sym_importance: importance,
    dqn.sym_episode_done: episode_done
}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print "================="
print "Test apply_op_train_td_():"
print "Initial td_loss: ",
print sess.run(dqn.sym_td_loss, feed_dict)
for i in range(10):
    dqn.apply_op_train_td_(
        state=state, action=action, reward=reward, next_state=next_state,
        next_action=next_action, episode_done=episode_done,
        importance=importance, sess=sess
    )
    print "td_loss after td step {}".format(i),
    print sess.run(dqn.sym_td_loss, feed_dict)

print "================="
print "Test apply_op_sync_target_():"
print "Initial diff l2: ",
print sess.run(dqn.sym_target_diff_l2)
for i in range(10):
    dqn.apply_op_sync_target_(sess)
    print "diff l2 after sync step {}".format(i),
    print sess.run(dqn.sym_target_diff_l2)
print 'pass!\n'

print "================="
print "Test improve value:"
print "Initial td_loss: ",
print sess.run(dqn.sym_td_loss, feed_dict)
for i in range(20):
    dqn.improve_value_(
        state, action, reward, next_state,
        next_action,
        episode_done,
        importance,
        sess
    )
    print "counters td {}, sync {}".format(dqn.countdown_td_, dqn.countdown_sync_),
    print "td_loss after td step {}".format(i),
    print sess.run(dqn.sym_td_loss, feed_dict),
    print "diff l2 after sync step {}".format(i),
    print sess.run(dqn.sym_target_diff_l2)
print 'pass!\n'

print "================="
print "Test get value:"

print "Case 1: batch:"
print dqn.get_value(state, action, sess=sess)
print "pass!\n"

print "Case 2: single sample (should raise exception):"
try:
    print dqn.get_value(state[0, :], action[0], sess=sess)
except Exception, error:
    print str(error)
finally:
    print "pass!\n"

print "Case 3: single sample with batch dim.:"
print dqn.get_value(state[0, :][np.newaxis, :], np.array(action[0, :])[np.newaxis, :], sess=sess)
print "pass!\n"

print "Case 4: default action (should raise exception):"
try:
    print dqn.get_value(state, sess=sess)
except Exception, error:
    print str(error)
finally:
    print "pass!\n"

print "=================="
print "Test get grad(q, action) by value:"
grad = dqn.get_grad_q_action(state, action, sess=sess)
print "Shape: {}".format(grad.shape)
print grad
print "pass!\n"

print "Test get grad(q_target, action) by value:"
grad = dqn.get_grad_q_action(state, action, sess=sess, use_target=True)
print "Shape: {}".format(grad.shape)
print grad
print "pass!\n"

sess.close()


