import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
#该语句会自动创建名为MNIST_data的文件夹，并下载MNIST数据集
#如果已存在，则直接读取数据集
mnist=input_data.read_data_sets("../Demo3/MNIST_data",one_hot=True)

#mini-batch size
batch_size=100

#训练集的数目
#print(mnist.train.num_examples)#55000
#一个epoch内包含的mini-batch个数
n_batch=mnist.train.num_examples // batch_size

#定义网络的输入和输出
x=tf.placeholder(tf.float32,[None,784])#28*28*1=784
y=tf.placeholder(tf.float32,[None,10])#0,1,2,3,4,5,6,7,8,9
keep_prob=tf.placeholder(tf.float32)

#创建一个的神经网络
#第一层
W1=tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1=tf.Variable(tf.zeros([2000])+0.1)
A1=tf.nn.tanh(tf.matmul(x,W1)+b1)
A1_drop=tf.nn.dropout(A1,keep_prob)
#第二层
W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2=tf.Variable(tf.zeros([2000])+0.1)
A2=tf.nn.tanh(tf.matmul(A1_drop,W2)+b2)
A2_drop=tf.nn.dropout(A2,keep_prob)
#第三层
W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3=tf.Variable(tf.zeros([1000])+0.1)
A3=tf.nn.tanh(tf.matmul(A2_drop,W3)+b3)
A3_drop=tf.nn.dropout(A3,keep_prob)
#输出层
W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(A3_drop,W4)+b4)

#学习率
lr=tf.Variable(0.001,dtype=tf.float32)

#交叉熵损失函数
# loss=tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#梯度下降法
#train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#统计预测结果
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#返回一个布尔型的列表
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        #学习率衰减
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})


        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("Iter "+str(epoch)+", Testing Accuracy "+str(test_acc)+", Training Accuracy "+str(train_acc))