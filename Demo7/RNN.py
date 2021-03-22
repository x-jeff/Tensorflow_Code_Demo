import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../Demo3/MNIST_data/", one_hot=True)

n_inputs = 28  # 输入层神经元个数,每个神经元代表图像的一行,一行为28个像素
max_time = 28  # 一个图像一共有28行
lstm_size = 100
n_classes = 10  # 10个分类:0~9
batch_size = 50  # 每批次50个样本
n_batch = mnist.train.num_examples // batch_size  # batch个数

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))  # 初始化权值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))  # 初始化偏置项


# 定义RNN网络
def RNN(X, weights, biases):
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state
    # final_state[1]是hidden state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    # results = tf.nn.softmax(tf.matmul(outputs[:, 27, :], weights) + biases)#验证下outputs和states之间的关系,这句和上一句结果是一样的
    return results


# 计算RNN的返回结果
prediction = RNN(x, weights, biases)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型
# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
