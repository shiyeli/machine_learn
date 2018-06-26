#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/25


import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 加载inference、train中的常量
import LeNet5_mnist_train as mnist_train
import LeNet5_mnist_inference as mnist_inference

# 每10秒加载一次最新的模型，并在测试数据上测试正确率
EVAL_INTERVAL_SECS = 50


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                [None, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS],
                name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        
        reshaped_images = np.reshape(mnist.validation.images, [
            mnist.validation.images.shape[0],
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS
        ])
        validate_feed = {x: reshaped_images, y_: mnist.validation.labels}
        
        y = mnist_inference.inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名方式来加载模型，这样在向前传播的过程中就不需要调用求滑动平均的函数来获取平均值了。
        # 这样就可以完全共用mnist_inference中定义的向前传播inference函数
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state会通过checkpoint文件自动找到目录中最新的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存是迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    
                    print 'After %s training steps,accuracy=%f' % (global_step, accuracy_score)
                else:
                    print 'No checkpoint file found'
                
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()