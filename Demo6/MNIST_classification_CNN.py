import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../Demo3/MNIST_data", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")
    y = tf.placeholder(tf.float32, [None, 10], name="y-input")
    with tf.name_scope("x_image"):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

with tf.name_scope("Conv1"):
    # 初始化第一个卷积层
    with tf.name_scope("w_conv1"):
        w_conv1 = weight_variable([5, 5, 1, 32], name="w_conv1")
    with tf.name_scope("b_conv1"):
        b_conv1 = bias_variable([32], name="b_conv1")

    # 卷积+pooling
    with tf.name_scope("conv2d_1"):
        conv2d_1 = conv2d(x_image, w_conv1) + b_conv1
    with tf.name_scope("relu"):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope("h_pool1"):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("Conv2"):
    # 初始化第二个卷积层
    with tf.name_scope("w_conv2"):
        w_conv2 = weight_variable([5, 5, 32, 64], name="w_conv2")
    with tf.name_scope("b_conv2"):
        b_conv2 = bias_variable([64], name="b_conv2")

    # 卷积+pooling
    with tf.name_scope("conv2d_2"):
        conv2d_2 = conv2d(h_pool1, w_conv2) + b_conv2
    with tf.name_scope("relu"):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope("h_pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("fc1"):
    # 初始化第一个全连接层
    with tf.name_scope("w_fc1"):
        w_fc1 = weight_variable([7 * 7 * 64, 1024], name="w_fc1")
    with tf.name_scope("b_fc1"):
        b_fc1 = bias_variable([1024], name="b_fc1")

    # 把池化层2的输出扁平化为1维
    with tf.name_scope("h_pool2_flat"):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name="h_pool2_flat")
    with tf.name_scope("wx_plus_b1"):
        wx_plus_b1 = tf.matmul(h_pool2_flat, w_fc1) + b_fc1
    with tf.name_scope("relu"):
        # 全连接层1
        h_fc1 = tf.nn.relu(wx_plus_b1)

    with tf.name_scope("keep_prob"):
        # dropout
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    with tf.name_scope("h_fc1_drop"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

with tf.name_scope("fc2"):
    # 全连接层2 / 输出层
    with tf.name_scope("w_fc2"):
        w_fc2 = weight_variable([1024, 10], name="w_fc2")
    with tf.name_scope("b_fc2"):
        b_fc2 = bias_variable([10], name="b_fc2")
    with tf.name_scope("wx_plus_b2"):
        wx_plus_b2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    with tf.name_scope("softmax"):
        prediction = tf.nn.softmax(wx_plus_b2)

with tf.name_scope("cross_entropy"):
    # cost function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train"):
    # 使用Adam优化算法
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

# 合并所有summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)
    for i in range(1001):
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})

        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)

        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print(
                "Iter " + str(i) + " , Testing Accuracy= " + str(test_acc) + " , Training Accuracy= " + str(train_acc))
