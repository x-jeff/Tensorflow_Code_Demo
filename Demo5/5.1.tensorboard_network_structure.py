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

#命名空间
with tf.name_scope("input"):
    #定义网络的输入和输出
    x=tf.placeholder(tf.float32,[None,784],name='x-input')#28*28*1=784
    y=tf.placeholder(tf.float32,[None,10],name='y-input')#0,1,2,3,4,5,6,7,8,9

with tf.name_scope("layer"):
    #创建一个简单的神经网络(无隐藏层)
    with tf.name_scope("weights"):
        W=tf.Variable(tf.zeros([784,10]),name="W")
    with tf.name_scope("biases"):
        b=tf.Variable(tf.zeros([10]),name="b")
    with tf.name_scope("wx_plus_b"):
        wx_plus_b=tf.matmul(x,W)+b
    with tf.name_scope("softmax"):
        prediction=tf.nn.softmax(wx_plus_b)

with tf.name_scope("loss"):
    #均方误差
    loss=tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope("train"):
    #梯度下降法
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.name_scope("accuracy"):
    #统计预测结果
    with tf.name_scope("correct_prediction"):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#返回一个布尔型的列表
    with tf.name_scope("accuracy"):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter("logs/",sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})


        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+", Testing Accuracy "+str(acc))