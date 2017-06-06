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
graph = tf.get_default_graph()

# Greedy policy
greedy_policy=False
print "=============="
print "Greedy: {}".format(greedy_policy)

dqn = DeepQFuncActionOut(
    gamma=0.99,
    f_net=f_net, state_shape=state_shape, num_actions=5,
    training_params=training_params, schedule=(2,5),
    greedy_policy=greedy_policy, graph=None
)

print "================="
print "Non-target subgraph in and out:"
print dqn.get_subgraph_value()

print "================="
print "Target subgraph in and out:"
print dqn.get_subgraph_value_target()

state = np.random.rand(10, 99 ,99, 3)
action = np.random.randint(0, 5, 10)
reward = np.random.rand(10)
next_state = np.random.rand(10, 99 ,99, 3)
next_action = np.random.randint(0, 5, 10)
importance = np.ones([10])
episode_done = np.zeros([10])

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
        sess, state, action, reward, next_state,
        next_action,
        importance,
        episode_done
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

print "================="
print "Test improve value:"
print "Initial improve_value(): ",
print sess.run(dqn.sym_td_loss, feed_dict)
for i in range(20):
    dqn.improve_value_(
        sess, state, action, reward, next_state,
        next_action,
        importance,
        episode_done
    )
    print "counters td {}, sync {}".format(dqn.countdown_td_, dqn.countdown_sync_),
    print "td_loss after td step {}".format(i),
    print sess.run(dqn.sym_td_loss, feed_dict),
    print "diff l2 after sync step {}".format(i),
    print sess.run(dqn.sym_target_diff_l2)    
sess.close()

