# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import pylab
mnist = input_data.read_data_sets(one_hot=True, train_dir='data/mnist')


def add_layer(inputs=None, in_size=1, out_size=1, opt_func=None):
    Weight = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + .5)
    Wx_b = tf.matmul(inputs, Weight) + biases
    if opt_func:
        outcome = opt_func(Wx_b)
    else:
        outcome = Wx_b
    return outcome


def compute_accuracy(my_xs, labels):
    global session, prediction_layer
    ys = session.run(prediction_layer, feed_dict={xs: my_xs})
    if_correct = tf.equal(tf.arg_max(ys, 1), tf.arg_max(labels, 1))
    accuarcy = tf.reduce_mean(tf.cast(if_correct, tf.float32))
    # labels are not needed to be added into feed_dict.
    # accuarcy doesn't contain this placeholder.
    result = session.run(accuarcy, feed_dict={xs: my_xs})
    return result


xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

prediction_layer = add_layer(xs, 784, 10, tf.nn.softmax)

cross_entropy = tf.reduce_mean(- tf.reduce_sum(ys * tf.log(prediction_layer), reduction_indices=[1]))
train_goal = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for step_ in xrange(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(train_goal, feed_dict={xs: batch_xs, ys: batch_ys})
        if step_ % 50 == 0:
            print step_, compute_accuracy(
                my_xs=mnist.test.images,
                labels=mnist.test.labels)
    one_pic_arr = np.reshape(batch_xs[0], (28, 28))
    plt.imshow(one_pic_arr)
    pylab.show()
    # Note that an 1-d np array is not a matrix. It needs to be 2-d.
    # Therefore I add [np.newaxis, :]
    batch_xs = batch_xs[0][np.newaxis, :]
    # Run the session once you need to handle a tensor.
    predict_result = session.run(prediction_layer, feed_dict={xs: batch_xs})
    output = tf.arg_max(predict_result, 1)
    print session.run(output, feed_dict={xs: batch_xs}) #[2]





