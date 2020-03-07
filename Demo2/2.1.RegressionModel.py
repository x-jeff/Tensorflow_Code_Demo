import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#使用numpy产生200个随机点作为训练数据
x_data=np.linspace(-0.5,0.5,200,axis=0).reshape(1,200)
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个placeholder
x=tf.placeholder(tf.float32,[1,None])
y=tf.placeholder(tf.float32,[1,None])

#定义神经网络中间层
W_L1=tf.Variable(tf.random_normal([10,1]))
b_L1=tf.Variable(tf.zeros([10,1]))
Z_L1=tf.matmul(W_L1,x)+b_L1
A_L1=tf.nn.tanh(Z_L1)

#定义神经网络输出层
W_L2=tf.Variable(tf.random_normal([1,10]))
b_L2=tf.Variable(tf.zeros(1,1))
Z_L2=tf.matmul(W_L2,A_L1)+b_L2
prediction=tf.nn.tanh(Z_L2)

#定义cost function：均方误差
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法求最小值
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})

    #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data[0,:],prediction_value[0,:],"r-",lw=5)
    plt.show()