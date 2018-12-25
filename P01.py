# coding:utf-8
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * .3 + .1

# variables

# A variable maintains state in the graph across calls to run().
# You add a variable to the graph by constructing an instance of the class Variable.
# 如果在 tensorflow 中设定了变量，那么初始化变量是最重要的
# 所以定义了变量以后, 一定要定义 init = tf.initialize_all_variables() .
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0, np.float32))
biases = tf.Variable(tf.zeros([1]))

y = x_data * weights + biases

# loss function & optimization
loss = tf.reduce_mean(tf.square(y - y_data))
optimization = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_goal = optimization.minimize(loss)

# init is a step, which needs to be run
init = tf.global_variables_initializer()

# run
session = tf.Session()
session.run(init)
for step_ in xrange(200):
    session.run(train_goal)
    if step_ % 20 == 0:
        print step_, session.run(weights), session.run(biases)

# To sum up:
# 1. Set train set: x_data and y_data
# 2. Set variables: weights and biases
# 3. Define loss function and optimization as a train_goal
# 4. Define init step
# 5. Define a session
# 6. Run init and train_goal in session.