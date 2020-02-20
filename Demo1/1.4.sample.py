import tensorflow as tf
import numpy as np

#使用numpy随机生成100个点
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2

#构造线性模型
b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

#最小二乘法
loss=tf.reduce_mean(tf.square(y_data-y))
#定义梯度下降法
optimizer=tf.train.GradientDescentOptimizer(0.2) #0.2为学习率
#最小化代价函数
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for iter in range(250):
        sess.run(train)
        if iter%20==0:
            print(iter,sess.run([k,b]))