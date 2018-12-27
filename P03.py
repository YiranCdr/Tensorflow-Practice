# coding:utf-8
import tensorflow as tf

# placeholder的值需要在Session.run()的时候以feed_dict参数传入
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# tf.multiply的乘法不是传统意义的矩阵乘法。
op = tf.matmul(input1, input2)

with tf.Session() as session:
    print session.run(op,
                      feed_dict={input1: [[1, 1]],
                                 input2: [[2], [3]]})

# [[5.]]
