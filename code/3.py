import pandas as pd
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

dataset = pd.read_csv('../data/fire_theft.csv', header=None, names=["x", "y"])

train_x = numpy.asarray(dataset["x"][1:], dtype=numpy.float)
train_y = numpy.asarray(dataset["y"][1:], dtype=numpy.float)

print(train_x, train_y)

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(numpy.random.ranf(), name="Weight")
b = tf.Variable(numpy.random.ranf(), name="bias")

pred = tf.add(tf.multiply(X, W), b)

size = len(dataset)

cost = tf.reduce_sum(tf.pow(Y-pred, 2))/size

learing_rate = 0.1
training_patches = 500
display_step = 50

optimizer = tf.train.GradientDescentOptimizer(learing_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for i in range(training_patches):
    for (x, y) in zip(train_x, train_y):
      sess.run(optimizer, feed_dict={X: x, Y: y})

    if (i + 1) % display_step == 0:
      c = sess.run(cost, feed_dict={X: x, Y: y})
      print(i + 1, sess.run(W), sess.run(b), c)

  print("Optimization Finished!")
  print("Y=", sess.run(W), "*X+", sess.run(b))
  cost_value = sess.run(cost, feed_dict={X: train_x, Y: train_y})
  print("Modal cost is:", cost_value)

  plt.plot(train_x, train_y, 'ro', label='Original data')
  plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
  plt.legend()
  plt.show()
