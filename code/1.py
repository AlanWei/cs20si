import tensorflow as tf

a = tf.add(3, 5)

with tf.Session() as sess:
  # print(sess.run(a))
  x = 2
  y = 3
  op1 = tf.add(x, y)
  op2 = tf.multiply(x, y)
  op3 = tf.pow(op1, op2)

  op3, a = sess.run([op3, a])
  print(op3, a)

sess.close();
