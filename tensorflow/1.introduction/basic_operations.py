import tensorflow as tf

a = tf.constant(2)

with tf.Session() as sess:
	print(sess.run(a))


a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

mul = tf.multiply(a,b)
add = tf.add(a,b)

with tf.Session() as sess:
	print(sess.run(mul, feed_dict={a:2, b:3}))
	print(sess.run(add, feed_dict={a:2, b:4}))


mul1 = tf.constant([[3, 3]])
mul2 = tf.constant([[2], [1]])

product1 = tf.matmul(mul1, mul2)
product2 = tf.matmul(mul2, mul1)

with tf.Session() as sess:
	print(sess.run(product2))
	print(sess.run(product1))