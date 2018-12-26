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
        # 此处的session.run并不是run(train_goal)，而是run(weights)
        # 因为weights是一个tensor，不可以直接输出，必须用run的形式看到
        # 不能直接输出的原因是tf采用静态图的机制，图中的信息只有run才能看到
        # 在tf中，tensor是一个线性操作。
        print step_, session.run(weights), session.run(biases)

session.close()

# To sum up:
# 1. Set train set: x_data and y_data
# 2. Set variables: weights and biases
# 3. Define loss function and optimization as a train_goal
# 4. Define init step
# 5. Define a session
# 6. Run init and train_goal in session.
# 7. Close session

# Output:
# 2018-12-25 23:26:36.122912:
# I tensorflow/core/platform/cpu_feature_guard.cc:141]
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 0 [-0.48678386] [0.7051269]
# 20 [0.0845153] [0.21306801]
# 40 [0.24637519] [0.12813774]
# 60 [0.2866551] [0.10700227]
# 80 [0.29667902] [0.10174257]
# 100 [0.29917353] [0.10043366]
# 120 [0.29979432] [0.10010793]
# 140 [0.29994884] [0.10002685]
# 160 [0.2999873] [0.10000668]
# 180 [0.29999685] [0.10000166]
