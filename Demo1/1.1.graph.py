import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #去除warning
tensor1 = tf.constant([[3,4]]) #定义一个constant类型的tensor，也是一个op
tensor2 = tf.constant([[5],[6]])
tensor3 = tf.matmul(tensor1,tensor2)
print("tensor1 : \n",tensor1)
print("tensor2 : \n",tensor2)
print("tensor3 : \n",tensor3)

#定义一个新的Session
sess = tf.Session()
#开始计算Graph
result = sess.run(tensor3)
print("result : \n",result)
#关闭Session
sess.close()

#以下方法可省略sess.close()
with tf.Session() as sess:
    result = sess.run(tensor3)
    print("result : \n",result)