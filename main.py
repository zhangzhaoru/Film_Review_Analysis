#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# zhangzhaoru
# 2019/7/10

import numpy as np
import time
import datetime
import os
import re
from random import randint
from os.path import isfile, join
import tensorflow as tf


"""
使用训练好的词向量模型,该矩阵包含400000x50的数据；还有一个是400000的词典
"""

wordsList = np.load('wordsList.npy')
print('载入word列表:',np.shape(wordsList),type(wordsList))
# 这里转化为列表,但是是[b'and', b'in', b'a', b'"', b"'s", b'for']的形式，即是二进制编码。
wordsList = wordsList.tolist()
# 转化为utf-8编码的形式，['of', 'to', 'and', 'in', 'a', '"', "'s", 'for', '-',]
wordsList = [word.decode('UTF-8')  for word in wordsList]
# 400000x50的嵌入矩阵，这个是训练好的词典向量模型
wordVectors = np.load('wordVectors.npy')
print('载入文本向量:',wordVectors.shape)
#print(wordVectors[home_ndex]) # 得到对应词典中词的50维向量


"""
解析文件，数据预处理，得到正负评价下的文件
"""

# 这里得到了所有的正面评价文件夹pos下的文件路径
pos_files = ['pos/' + f
             for f in os.listdir('pos/')
                 if isfile(join('pos/', f))]
# 这里得到了所有的neg文件夹下的文件路径
neg_files = ['neg/' + f
             for f in os.listdir('neg/')
                 if isfile(join('neg/', f))]
num_words = []
# 这里的每个txt文件都是一行文本，12500都是正面评价
for pf in pos_files:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('正面评价完结。。。')
# 这里是12500负面评价
for nf in neg_files:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('负面评价完结。。。')
num_files = len(num_words)
print('文件总数:', num_files)
print('所有的词的数量:', sum(num_words))
print('平均文件的词的长度:', sum(num_words) / len(num_words))


"""
辅助函数：返回一个数据集的迭代器，用于返回一批训练集合

"""
max_seq_num = 250
num_dimensions = 50  # 每个单词向量的维度，这里和嵌入矩阵的每个词的维度相同

# arr:24 x 250的矩阵
def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels

# 同上
def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


"""
构建tensorflow图

"""

batch_size = 24 # batch的尺寸
lstm_units = 64 # lstm的单元数量
num_labels = 2  # 输出的类别数
iterations = 200000 # 迭代的次数
# 载入正负样本的词典映射
ids = np.load('idsMatrix.npy')
print('载入IDS:',ids.shape)
tf.reset_default_graph()
# 确定好单元的占位符：输入是24x300，输出是24x2
labels = tf.placeholder(tf.float32, [batch_size, num_labels])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])

# 必须先定义该变量
data = tf.Variable(
    tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)
# 调用tf.nn.lookup()接口获得文本向量，该函数返回batch_size个文本的3D张量，用于后续的训练
data = tf.nn.embedding_lookup(wordVectors, input_data)

# 使用tf.contrib.rnn.BasicLSTMCell细胞单元配置lstm的数量
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
# 配置dropout参数，以此避免过拟合
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
# 最后将LSTM cell和数据输入到tf.nn.dynamic_rnn函数，功能是展开整个网络，并且构建一整个RNN模型
# 这里的value认为是最后的隐藏状态，该向量将重新确定维度，然后乘以一个权重加上偏置，最终获得lable
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

# 定义正确的预测函数和正确率评估参数
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# 最后将标准的交叉熵损失函数定义为损失值，这里是以adam为优化函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


tf.summary.scalar('loss',loss)
tf.summary.scalar('Accrar',accuracy)
merged=tf.summary.merge_all()
logdir='tensorboard/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"
writer=tf.summary.FileWriter(logdir,sess.graph)


iterations = 100000
for i in range(iterations):
    # 下个批次的数据
    next_batch, next_batch_labels = get_train_batch()
    sess.run(optimizer,{input_data: next_batch, labels: next_batch_labels})

    # 每50次写入一次leadboard
    if(i%50==0):
        summary=sess.run(merged,{input_data: next_batch, labels: next_batch_labels})
        writer.add_summary(summary,i)

    if (i%1000==0):
        loss_ = sess.run(loss, {input_data: next_batch, labels: next_batch_labels})

        accuracy_=(sess.run(accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100
        print("iteration:{}/{}".format(i+1, iterations),
                  "\nloss:{}".format(loss_),
                  "\naccuracy:{}".format(accuracy_))
        print('..........')
    # 每10000次保存一下模型
    if(i%10000==0 and i!=0):
        save_path=saver.save(sess,"models/pretrained_lstm.ckpt",gloal_step=i)
        print("saved to %s"% save_path)

writer.close()
