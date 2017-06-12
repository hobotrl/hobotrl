import sys
sys.path.append('../../../')

import numpy as np

import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer

from hobotrl.tf_dependent.value_function import DeepQFuncActionOut

def f_net(inputs, num_outputs, is_training):
    depth = inputs.get_shape()[1:].num_elements()
    inputs = tf.reshape(inputs, shape=[-1, depth])
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
        inputs=hidden2, units=num_outputs,
        activation=tf.nn.tanh,
        kernel_regularizer=l2_regularizer(scale=1e-4),
        trainable=True, name='out',
    )
    return q

optimizer_td = tf.train.GradientDescentOptimizer(learning_rate=0.001)
target_sync_rate = 0.01
training_params = (optimizer_td, target_sync_rate)
state_shape = (99, 99, 3)
batch_size = 10
num_actions = 5
graph = tf.get_default_graph()

# Initialization
greedy_policy = True
ddqn = True
print "=============="
print "Test initialize QFunc with greedy_policy={} and ddqn={}:".format(
    greedy_policy, ddqn
),

dqn = DeepQFuncActionOut(
    gamma=0.99,
    f_net_dqn=f_net, state_shape=state_shape, num_actions=num_actions,
    training_params=training_params, schedule=(2,5),
    batch_size=batch_size,
    greedy_policy=greedy_policy, ddqn=ddqn,
    graph=None
)
print 'pass!'
raw_input('next test case? <enter>')

print "================="
print "Non-target subgraph in and out:"
print dqn.get_subgraph_value()
print 'pass!'
raw_input('next test case? <enter>')

print "================="
print "Target subgraph in and out:"
print dqn.get_subgraph_value_target()
print 'pass!'
raw_input('next test case? <enter>')

state = np.random.rand(batch_size, 99 ,99, 3)
action = np.random.randint(0, num_actions, batch_size)
reward = np.random.rand(batch_size)
next_state = np.random.rand(batch_size, 99 ,99, 3)
next_action = np.random.randint(0, num_actions, batch_size)
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
print "pass!"
raw_input('next test case? <enter>')

print "================="
print "Test apply_op_sync_target_():"
print "Initial diff l2: ",
print sess.run(dqn.sym_target_diff_l2)
for i in range(10):
    dqn.apply_op_sync_target_(sess)
    print "diff l2 after sync step {}".format(i),
    print sess.run(dqn.sym_target_diff_l2)
print 'pass!'
raw_input('next test case? <enter>')

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
print 'pass!'
raw_input('next test case? <enter>')

print "================="
print "Test get value:"

print "Case 1: batch:"
print dqn.get_value(state, action, sess=sess)
print "pass!"
raw_input('next test case? <enter>')

print "Case 2: single sample (should raise exception):"
try:
    print dqn.get_value(state[0, :], action[0], sess=sess)
except Exception, error:
    print str(error)
finally:
    print "pass!"
raw_input('next test case? <enter>')

print "Case 3: single sample with batch dim.:"
print dqn.get_value(state[0, :][np.newaxis, :], np.array(action[0])[np.newaxis], sess=sess)
print "pass!"

print "Case 4: default action:"
print dqn.get_value(state, sess=sess)
print "pass!"

sess.close()


