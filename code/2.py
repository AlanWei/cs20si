import tensorflow as tf

a = tf.constant(3, name="a")
b = tf.constant(5, name="b")

x = tf.add(a, b, name="add")

with tf.Session() as sess:
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  print sess.run(x)
  print sess.run(tf.zeros([1, 2])), sess.run(tf.ones([2, 2]))

writer.close()
