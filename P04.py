# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# define add layer
def add_layer(input, input_size, output_size, activation_function=None):
    Weights = tf.Variable(tf.random_uniform([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    Wx_b = tf.matmul(input, Weights) + biases
    if activation_function:
        output = activation_function(Wx_b)
    else:
        output = Wx_b
    return output


# 300个训练样例(不是三百个神经元的输入层)
# shape(x_data) = (3, )
x_data = np.linspace(start=-1, stop=1, num=300)
# shape(x_data) = (3, 1)
x_data = x_data[:, np.newaxis]
noise = np.random.normal(loc=0, scale=0.05, size=x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# placeholder可以理解为数据容器的说
xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

layer1 = add_layer(xs, 1, 10, tf.nn.relu)
output_layer = add_layer(layer1, 10, 1, None)

loss = tf.reduce_mean(tf.reduce_sum((tf.square(output_layer - ys)), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_goal = optimizer.minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
# 非block的持续输出
plt.ion()
plt.show()
with tf.Session() as session:
    session.run(init)
    for step_ in xrange(2000):
        session.run(train_goal, feed_dict={xs: x_data, ys: y_data})
        if step_ % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            y_predict = session.run(output_layer, feed_dict={xs: x_data})
            print step_, session.run(loss, feed_dict={xs: x_data, ys: y_data})
            lines = ax.plot(x_data, y_predict, 'r-', lw=5)
            plt.pause(1)


