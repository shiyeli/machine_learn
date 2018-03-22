#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> on 2018/2/27 10:50


#http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#占位符
x=tf.placeholder(tf.float32,shape=[None,784])
#变量
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))



# 向前计算
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_=tf.placeholder(tf.float32,shape=[None,10])
# 交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(50)
	sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})
