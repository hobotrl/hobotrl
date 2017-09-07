import hobotrl as hrl
import tensorflow as tf
import hobotrl.network as network
from initialD_input import distorted_inputs

def f_net(inputs):
    l2 = 1e-4
    state = inputs[0]
    conv = hrl.utils.Network.conv2ds(state, shape=[(32, 4, 4), (64, 4, 4), (64, 2, 2)], out_flatten=True,
                                     activation=tf.nn.relu,
                                     l2=l2, var_scope="convolution")
    q = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
                                    l2=l2, var_scope="q")
    pi = hrl.network.Utils.layer_fcs(conv, [200, 100], 3,
                                     activation_out=tf.nn.softmax, l2=l2, var_scope="pi")
    return {"q": q, "pi": pi}

def init_network(f_create_net, state_shape):

    return

def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr

# construct network
state_shape = (224, 224, 3)
x = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_shape), name="image")
networks = network.Network([x], f_net, var_scope="learn")
q = networks['q']
pi = networks['pi']

# construct loss and train operator
y_ = tf.placeholder(dtype=tf.int32, shape=[None]+3, name="action")
cross_entropy = -tf.reduce_mean(y_*tf.log(pi))
train_op = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

# set hyper parameters
logdir = "./tmp_sltrain"
train_dataset = None
batch_size = 64
lr_decay_steps = 500
max_step = 2000
initial_lr = 0.01
lr_decay = 0.01

# set session parameters
init_op = tf.global_variables_initializer()
graph = tf.get_default_graph()
global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False
        )

sv = tf.train.Supervisor(graph=graph,
                        global_step=global_step,
                        init_op=init_op,
                        summary_op=None,
                        summary_writer=None,
                        logdir=logdir,
                        save_summaries_secs=0)

with sv.managed_session() as sess:
    train_images, train_labels = distorted_inputs(train_dataset, batch_size, num_threads=6)
    # =============== problem==============
    init_step = global_step.eval(sess=sess)
    tf.train.start_queue_runners(sess)
    for step in xrange(init_step, max_step):
        # lr_value = get_lr(initial_lr, lr_decay, lr_decay_steps, global_step)
        np_train_images, np_train_labels = sess.run([train_images, train_labels])
        sess.run(train_op, feed_dict={x:np_train_images, y_:np_train_labels})
        if step % 500 == 0:
            sv.saver.save(sess, logdir=logdir, global_step=step)