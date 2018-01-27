# 实现卷积层和池化层功能的小程序
import tensorflow as tf
import numpy as np

# 定义输入的矩阵,注意输出的维度，从而了解到矩阵维度的嵌套
M = np.array([
    [[1], [-1], [0]],
    [[-1], [2], [1]],
    [[0], [2], [-2]]
])

# 该矩阵只是用来对比矩阵输出维度用的，不做与后面卷积与此话操作
M1 = np.array([
    [1, -1, 0],
    [-1, 2, 1],
    [0, 2, -2]
])
print("Matrix shape is: ", M.shape)
print("Matrix shape is: ", M1.shape)

# 定义卷积过滤器尺寸为2*2，当前深度为1,下一层深度（即过滤器深度）为1
filter_weight = tf.get_variable('weight', [2, 2, 1, 1], initializer=tf.constant_initializer([[1, -1],
                                                                                             [0, 2]]))

# 参数[1]（如果假设为16）代表的是总共有下一层深度个不同偏置项，
# 1（如果假设为16）为过滤器的深度，也是神经网络下一层节点矩阵的深度
# 详细参考书本146页《Tensorflow-google》
biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(1))

# 调整输入的格式符合Tensorflow的要求
M = np.asanyarray(M, dtype='float32')
# 调整矩阵维度
M = M.reshape(1, 3, 3, 1)

# 计算矩阵通过卷积层过滤器和池化层过滤器计算后的结果
x = tf.placeholder('float32', [1, None, None, 1])
# 依次为:(卷积-加偏置)-池化，其中在这个卷积里面的步长为2
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding='SAME')
bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

# 创建一个会话来运行
with tf.Session() as sess:
    # 下面的两句初始化可以换成tf.global_variables_initializer().run()
    INIT = tf.global_variables_initializer()
    sess.run(INIT)
    convoluted_M = sess.run(bias, feed_dict={x: M})
    pooled_M = sess.run(pool, feed_dict={x: M})

    print("convoluted_M: \n", convoluted_M)
    print("pooled_M: \n", pooled_M)
