#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/22




import LeNet5_mnist_inference as mnist_inference
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.05
LEARNING_RATE_DECAY = 0.3
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './tmp/LeNet5/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(
        tf.float32,
        [None, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS],
        name='x-input'
    )
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # 向前传播
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, [
                BATCH_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.NUM_CHANNELS
            ])
            _, loss_value, step ,_learning_rate= sess.run([train_op, loss, global_step,learning_rate], feed_dict={x: reshaped_xs, y_: ys})

            # 每1000轮保存一次模型
            if (i+1) % 500 == 0:
                print 'After %d training steps,loss=%f,learning_rate=%f' % (step, loss_value,_learning_rate)

                # 保存模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)


def main(argv=None):
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    """
    ValueError: Variable layer1/weights already exists, disallowed.
    Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
    第一次运行将下行代码注释，以后运行放开
    """
    # tf.get_variable_scope().reuse_variables()

    tf.app.run()


"""
After 500 training steps,loss=0.878618,learning_rate=0.028958
After 1000 training steps,loss=0.853497,learning_rate=0.016753
After 1500 training steps,loss=0.801199,learning_rate=0.009692
After 2000 training steps,loss=0.705418,learning_rate=0.005607
After 2500 training steps,loss=0.719921,learning_rate=0.003244
After 3000 training steps,loss=0.685588,learning_rate=0.001877
After 3500 training steps,loss=0.731382,learning_rate=0.001086
After 4000 training steps,loss=0.632257,learning_rate=0.000628
After 4500 training steps,loss=0.700597,learning_rate=0.000363
After 5000 training steps,loss=0.659599,learning_rate=0.000210
After 5500 training steps,loss=0.721710,learning_rate=0.000122
After 6000 training steps,loss=0.663642,learning_rate=0.000070
After 6500 training steps,loss=0.689002,learning_rate=0.000041
After 7000 training steps,loss=0.644999,learning_rate=0.000024
After 7500 training steps,loss=0.660535,learning_rate=0.000014
After 8000 training steps,loss=0.753764,learning_rate=0.000008
After 8500 training steps,loss=0.775101,learning_rate=0.000005
After 9000 training steps,loss=0.762689,learning_rate=0.000003
After 9500 training steps,loss=0.871014,learning_rate=0.000002
After 10000 training steps,loss=0.655120,learning_rate=0.000001


After 10000 training steps,accuracy=0.982400
"""