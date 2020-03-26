import tensorflow as tf

vector=[1,2,3,10,6,7,8]
matrix=[[1,2,3,5],[2,8,4,6],[10,2,5,7]]

with tf.Session() as sess:
    idx1=tf.argmax(vector,0)
    idx2=tf.argmax(matrix,0)
    idx3=tf.argmax(matrix,1)

    print(sess.run(idx1))
    print(sess.run(idx2))
    print(sess.run(idx3))