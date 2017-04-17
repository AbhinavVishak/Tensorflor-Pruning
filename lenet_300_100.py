from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers.regularizers import l1_regularizer
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer

# Import data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])

n_hidden_1 = 300  # 1st layer num features
n_hidden_2 = 100  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

W = {
    'fc1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'fc2':
    tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}

biases = {
    'fc1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}


def model(_X, _W, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _W['fc1']), _biases['fc1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _W['fc2']), _biases['fc2']))
    return tf.matmul(layer_2, _W['out']) + _biases['out']


def l1(x=0.0001):
    regularizers = (
        l1_regularizer(.1)(W['fc1']) + l1_regularizer(.1)(biases['fc1']))
    regularizers += (
        l1_regularizer(.1)(W['fc2']) + l1_regularizer(.1)(biases['fc2']))
    regularizers += (
        l1_regularizer(.1)(W['out']) + l1_regularizer(.1)(biases['out']))
    regularizers = x * regularizers
    return regularizers


l = model(x, W, biases)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

correct_prediction = tf.equal(tf.argmax(l, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=l))

cost = cross_entropy + l1()

train_step = tf.train.AdamOptimizer(
    0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
    use_locking=False).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model

print(
    sess.run(
        accuracy, feed_dict={x: mnist.test.images,
                             y_: mnist.test.labels}))


def prune(x):
    y_noprune = sess.run(x)
    y_noprune = np.asarray(y_noprune)
    low_values_indices = abs(y_noprune) < 0.04
    y_prune = y_noprune
    y_prune[low_values_indices] = 0
    return y_noprune, y_prune


w_fc1_, w_fc1 = prune(W['fc1'])

W_prune = {}
W_prune['fc1'] = W['fc1'].assign(w_fc1, use_locking=False)

w_fc2_, w_fc2 = prune(W['fc2'])
W_prune['fc2'] = W['fc2'].assign(w_fc2, use_locking=False)

w_out_, w_out = prune(W['out'])
W_prune['out'] = W['out'].assign(w_out, use_locking=False)

sess.run(W_prune)

print("sparsity of w_fc1=",
      float(np.count_nonzero(w_fc1)) / float(np.size(w_fc1)) * 100, "%")
print("sparsity of w_fc2=",
      float(np.count_nonzero(w_fc2)) / float(np.size(w_fc2)) * 100, "%")
print("sparsity of w_out=",
      float(np.count_nonzero(w_out)) / float(np.size(w_out)) * 100, "%")

print(
    sess.run(
        accuracy, feed_dict={x: mnist.test.images,
                             y_: mnist.test.labels}))

print(W)
print(W_prune)
#saver = tf.train.Saver()
#print(saver.save(sess, save_path='./savedmodel'))
