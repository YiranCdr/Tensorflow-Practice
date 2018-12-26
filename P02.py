import tensorflow as tf

x = tf.constant([[-2.25 + 4.75], [-3.25 + 5.75]])
a = tf.abs()
print a
sess = tf.Session()
print sess.run(a)
