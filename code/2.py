import tensorflow as tf

a = tf.constant(3, name="a")
b = tf.constant(5, name="b")

x = tf.add(a, b, name="add")

zeros = tf.zeros([2, 2])
ones = tf.ones([2, 2])
fill = tf.fill([2, 2], 8)
# n number with equal distance between start and end
linspace = tf.linspace(10.0, 13.0, 5)
# exclude end
ran = tf.range(1, 5, 1)
addN = tf.add_n([1, 2, 3])

my_const = tf.constant([1.0, 2.0], name="my_const")

test = tf.zeros([3])
placeholder = tf.placeholder(tf.float32, shape=[3], name=None)

operation = test + placeholder

with tf.Session() as sess:
  # writer = tf.summary.FileWriter('./graphs', sess.graph)
  print sess.run(operation, feed_dict={placeholder: [1, 2, 3]})

# writer.close()
