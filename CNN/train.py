#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import t_wiki_process

# 设置参数的过程
# Parameters
# ==================================================

# Data loading params
# 注释参考这篇文章，http://blog.csdn.net/github_38414650/article/details/74019595
# tf.flags.DEFINE_float/string(flag_name, default_value, docstring(文档字符串)), 对该函数的理解是把原始文件的名字重新命名为data_helps里面所需要的文件名
#  其次该函数带有输出的功能？？？？
# 数据集里10%为验证集；POS正例；NEG反例
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")   # 10%交叉验证集
tf.flags.DEFINE_string("train_data", "D:\\tensorflow\Atr_fortest.txt", "经过提取后只含全部属性的数据.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg-1", "Data source for the negative data.")

# Model Hyperparameters
# embedding维度128，3种卷积核(3,4,5)，每种128个，0.5的dropout；
# 参数经过改动，是否和默认参数不同,比如如果是3.4.5，为何运行结果不对？
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
# batch_size：1次迭代所使用的样本量； ；一个epoch是指把所有训练数据完整的过一遍；iteration：表示1次迭代，每次迭代更新1次网络结构的参数
tf.flags.DEFINE_integer("batch_size", 5, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
# true表示自动寻找一个存在并支持的cpu或者gpu，防止指定的设备不存在
# 如果将False改为True，可以看到operations被指派到哪个设备运行
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# ？？？？？？？
#FLAGS是一个对象，保存了解析后的命令行参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================
print("time"+"\t\t"+str(datetime.datetime.now().isoformat()))

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels_QA(FLAGS.train_data)
print(y)
print("time"+"\t\t"+str(datetime.datetime.now().isoformat()))

# Build vocabulary
##########################################################
#  在这一步把字符串表示的单词，参照wiki语料库词向量生成输入 #
##########################################################
##########################################################
#     利用建立一个词汇表（字典形式） ，把字变成索引矩阵     #
##########################################################
max_document_length = max([len(x) for x in x_text])  # 获取单行的最大的长度
print("max_document_length:",max_document_length)
dir = t_wiki_process.get_wiki_dic("D:\\tensorflow\wiki.vector")
vocab_dir = {}

# 建立词汇表（字典形式，｛字：number｝）
# 词汇表从1开始，因为建立单词索引表不够长的填0，所以从1开始
count = 0
for key in dir:
    vocab_dir[key] = count + 1
    count += 1

# 建立单词索引表
all_index  = []
for word in x_text:
    index =[]
    size = len(word)
    for i in word:
        number = vocab_dir[i]
        index.append(number)
        #print(index)
    if size != max_document_length:
        dif = (max_document_length - size)
        for j in range(dif):
            index.append(0)
    all_index.append(index)
x = np.array(all_index)

print("##########################")
print(x.shape)
print(x)
print("##########################")
print("time"+"\t\t"+str(datetime.datetime.now().isoformat()))

# Randomly shuffle data
np.random.seed(10)    # 运用seed使得每次生成的随机数相同
shuffle_indices = np.random.permutation(np.arange(len(y)))  # permutation打乱样本,且有返回值，而假设使用shuffle则没有返回值
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
print("time"+"\t\t"+str(datetime.datetime.now().isoformat()))

# Split train/test set
# 分割训练集和测试集
# TODO: This is very crude(粗糙), should use cross-validation（交叉验证）
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))   #????
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# 删除以下的四个变量,但是数据还在并没有删除   http://blog.csdn.net/love1code/article/details/47276683
del x, y, x_shuffled, y_shuffled

#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))    #vocabulary_????这是才用预处理数据的意思？？
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# y_train的输出如下
#[[1 0]
# [1 0]
# [1 0]
# ...,
# [0 1]
# [0 1]
# [1 0]]
# y_train载入的是data_helpers里面的标签数据,维度是（9596，2）.


# Training
# ==================================================

# tf.Graph----http://blog.csdn.net/zj360202/article/details/78539464
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)  # 这个是session的配置过程，按照前面的gpu，cpu自动选择

    sess = tf.Session(config=session_conf)  # 建立一个配置如上的会话
    with sess.as_default():
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(len(x_train))
        print(x_train)
        print(x_train.shape)
        print(x_train.shape[0])
        print(x_train.shape[1])
        print(y_train)
        print(y_train.shape)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # 从text_cnn.py中导入进TextCNN类
        cnn = TextCNN(
            # shape[0]就是读取矩阵第一维度的长度
            # shape[1]就是读取矩阵第二维度的长度
            sequence_length=x_train.shape[1],  # x_train.shape[1]句子的个数，x_train.shape[0]样本的个数
                                               # 单个句子的最大长度（1个句子中单词的个数） max_document_length
            num_classes=y_train.shape[1],      # 分类的种类0,1，在这就是2
            index = x_train,
            embedding_size=FLAGS.embedding_dim, # 向量化的维度128
            filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),  # 卷积层的层数3
            num_filters=FLAGS.num_filters,  # 卷积核个数128
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)  # 全局步骤
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        # 保存日志以便可视化的相关操作，相关可看http://blog.csdn.net/helei001/article/details/51842531
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)  # histogram直方图
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))   # scalar数量/标量
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))   # os.path.curdir当前目录，即在当前目录加入runs文件夹保存日志和模型
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)    #正式写入日志

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # 保存模型的部分
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):    #训练集
            """
            A single training step
            """
            # 定义这个字典和下面的sess.run是啥子意思??
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        # zip():将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # 参见https://www.cnblogs.com/waltsmith/p/8029539.html
        # 这里data_helpers.batch_iter返回的是一个矩阵
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)  # zip（*）反解压,参见网页如上
            #x_batch = np.array(x_batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)  # 计算步骤总数
            if current_step % FLAGS.evaluate_every == 0:  # 每evaluate_every步输出一次验证集的结果，此处为每五次
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)  # 用验证集作为输入
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)  # 同理，此处每十次保存一次模型
                print("Saved model checkpoint to {}\n".format(path))
