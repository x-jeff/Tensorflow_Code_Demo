import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #去除warning

#Fetch
c1=tf.constant(1)
c2=tf.constant(2)
c3=tf.constant(3)
add1=tf.add(c2,c3)
mul1=tf.multiply(c1,add1)
with tf.Session() as sess:
    result1=sess.run([mul1,add1])
    print(result1)

#Feed
p1=tf.placeholder(tf.float32)
p2=tf.placeholder(tf.float32)
mul2=tf.multiply(p1,p2)
with tf.Session() as sess:
    result2=sess.run(mul2,feed_dict={p1:[5.],p2:[.3]})
    # result2=sess.run(mul2,feed_dict={p1:5.,p2:.3})
    print(result2)
