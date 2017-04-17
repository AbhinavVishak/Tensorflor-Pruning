import tensorflow as tf

sess = tf.Session()
a = tf.train.import_meta_graph('./savedmodel.meta')
a.restore(sess, './savedmodel')
z = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
print(z[1].name + '\n')
print(sess.run(z[1]))