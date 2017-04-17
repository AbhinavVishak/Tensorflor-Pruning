import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
a = tf.train.import_meta_graph('./savedmodel.meta')

sess.run(tf.global_variables_initializer())
print(
    sess.run(
        'acc:0',
        feed_dict={'x:0': mnist.test.images,
                   'y_:0': mnist.test.labels}))

a.restore(sess, './savedmodel')
print(
    sess.run(
        'acc:0',
        feed_dict={'x:0': mnist.test.images,
                   'y_:0': mnist.test.labels}))


def prune(x, t=0.03):
    y_noprune = sess.run(x)
    y_noprune = np.asarray(y_noprune)
    low_values_indices = abs(y_noprune) < t
    y_prune = y_noprune
    y_prune[low_values_indices] = 0
    if x == 'fc1:0':
        tz = y_prune.flatten()
        bins = np.linspace(np.amin(tz) - 0.01, np.amax(tz) + 0.01, 256)
        qwe = np.digitize(y_prune, bins)
        #y_prune = bins[qwe]
    return y_noprune, y_prune


W_prune = {}

temp = tf.get_default_graph().get_tensor_by_name
w_fc1_, w_fc1 = prune('fc1:0')
W_prune['fc1'] = tf.assign(temp('fc1:0'), w_fc1, use_locking=False)

w_fc2_, w_fc2 = prune('fc2:0')
W_prune['fc2'] = tf.assign(temp('fc2:0'), w_fc2, use_locking=False)

w_out_, w_out = prune('out:0')
W_prune['out'] = tf.assign(temp('out:0'), w_out, use_locking=False)

sess.run(W_prune)

print("sparsity of w_fc1=",
      float(np.count_nonzero(w_fc1)) / float(np.size(w_fc1)) * 100, "%")
print("sparsity of w_fc2=",
      float(np.count_nonzero(w_fc2)) / float(np.size(w_fc2)) * 100, "%")
print("sparsity of w_out=",
      float(np.count_nonzero(w_out)) / float(np.size(w_out)) * 100, "%")

print(
    sess.run(
        'acc:0',
        feed_dict={'x:0': mnist.test.images,
                   'y_:0': mnist.test.labels}))
