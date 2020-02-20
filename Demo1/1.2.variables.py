import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #去除warning
v1 = tf.Variable([1,2])
c1 = tf.constant([3,4])
sub = tf.subtract(v1,c1) #定义一个减法op
add = tf.add(v1,c1) #定义一个加法op

init = tf.global_variables_initializer() #初始化全局变量

with tf.Session() as sess:
    #sess.run(init)
    sess.run(v1.initializer)
    sub_result = sess.run(sub)
    add_result = sess.run(add)
    print(sub_result)
    print(add_result)

v2 = tf.Variable(0,name="counter")
new_v2 = tf.add(v2,1)
update_v2 = tf.assign(v2,new_v2) #用于赋值操作
init = tf.global_variables_initializer() #初始化全局变量
with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(update_v2)
        # print(sess.run(update_v2))
        print(sess.run(v2))