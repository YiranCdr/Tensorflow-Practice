# coding:utf-8
import tensorflow as tf
import numpy as np


# define add layer
def add_layer(layer_name, input, input_size, output_size, activation_function=None):
    with tf.name_scope(layer_name):
        with tf.name_scope('Weight'):
            Weights = tf.Variable(tf.random_uniform([input_size, output_size]))
            tf.summary.histogram('Weight', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
            tf.summary.histogram('biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_b = tf.matmul(input, Weights) + biases
        with tf.name_scope('activation_function'):
            if activation_function:
                output = activation_function(Wx_b)
            else:
                output = Wx_b
            tf.summary.histogram('output', output)
        return output


# 300个训练样例(不是三百个神经元的输入层)
# shape(x_data) = (3, )
x_data = np.linspace(start=-1, stop=1, num=300)
# shape(x_data) = (3, 1)
x_data = x_data[:, np.newaxis]
noise = np.random.normal(loc=0, scale=0.05, size=x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# placeholder可以理解为数据容器的说
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, name='x_input')
    ys = tf.placeholder(tf.float32, name='y_input')

layer1 = add_layer('hidden_layer', xs, 1, 10, tf.nn.relu)
output_layer = add_layer('output_layer', layer1, 10, 1, None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum((tf.square(output_layer - ys)), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_goal = optimizer.minimize(loss)

init = tf.global_variables_initializer()

session = tf.Session()
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir='log/', graph=session.graph)
session.run(init)

for step_ in xrange(1000):
    session.run(train_goal, feed_dict={xs: x_data, ys: y_data})
    if step_ % 50 == 0:
        # run merge
        result = session.run(merge, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, step_)
