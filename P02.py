import tensorflow as tf

state = tf.Variable('hello_world')
update = tf.add(state, '!')

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print session.run(update)
