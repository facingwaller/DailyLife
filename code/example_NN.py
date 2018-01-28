import tensorflow as tf

# Numpy是一个科学计算工具包，这里通过Numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数,stddev为标准差
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))


# 在shape的一个维度上使用None可以方便使用不大的batch大小。在训练的时候需要把数据分成
# 较小的batch，但是在测试时，可以一次性的使用全部数据。当数据集比较小时这样比较方便
# 测试，但是数据集比较大的时候，将大量数据放入batch可能会导致内存溢出。
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络的前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法。
# 损失函数
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# 反向传播算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集。
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义规则来给出样本的标签。在这里x1+x2<1的样例就认为是正样本（比如零件合格），
# 而其他为负样本（比如零件不合格）。而在这，具体而言，就是0为负样本，1为正样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行Tensorflow程序。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    """    在训练之前神经网络参数的值
       w1=[[-0.81131822,1.48459876,0.06532937]
           [-2.44270396,0.0992484,0.59122431]]
       w2=[[-0.81131822],[1.48459876],[0.06532937]] 
   """



# 设定训练的轮数。
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练。
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出。
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y}
            )
            print("After %d training step(s), cross entropy on all data is %g"%(i, total_cross_entropy))
    """ 
        输出结果：
        After 0 training step(s), cross entropy on all data is 0.0674925
        After 1000 training step(s), cross entropy on all data is 0.0163385
        After 2000 training step(s), cross entropy on all data is 0.00907547
        After 3000 training step(s), cross entropy on all data is 0.00714436
        After 4000 training step(s), cross entropy on all data is 0.00578471

        通过这个结果可以发现随着训练的进行，交叉熵逐渐变小，说明预测结果和真实结果差距越小
    """

    print(sess.run(w1))
    print(sess.run(w2))
"""
w1 = [[-1.9618274, 2.58235407, 1.68203783]
      [-3.4681716, 1.06982327, 2.11788988]]
w2 = [[-1.8247149], [2.68546653], [1.41819501]]

可以发现俩个参数的取值已经发生了变化，这就是训练的结果。它使得这个神经网络可以更好的拟合提供的训练数据。
"""