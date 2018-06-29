#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/28


import inception_v3_transfer_learn_datasets as datasets
import inception_v3_transfer_learn_inferance as inferance
import tensorflow as tf
import numpy as np

LEARNING_RATE=0.00001
STEPS=1000
BATCH=50
MODE_SAVE_PATH='tmp/transfer/model.ckpt'
TEST_IMAGES=500


def train(datasets,test_datasets):
    
    # 输入变量初始化
    x = tf.placeholder(tf.float32, [None,inferance.POOL_3_RESHAPE_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inferance.NUM_CLASS], name='y-input')

    # 向前传播
    y = inferance.inference_custom(x)
    global_step = tf.Variable(0, trainable=False)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,
            labels=tf.argmax(y_, 1)
    )
    
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(STEPS):
            xs, ys = datasets.next_batch(BATCH)
            sess.run(train_step, feed_dict={x : xs,y_: ys })
            
            if i % 100 == 0:
                loss_eval = sess.run(loss, feed_dict={x: xs, y_: ys})
                accuracy_score = sess.run(accuracy,feed_dict={x: test_datasets.datasets[0], y_: test_datasets.datasets[1]})
                print 'Step:%d, loss=%f, accuracy=%f' % (i, loss_eval,accuracy_score)
                
            # 保存模型
            # saver.save(sess, MODE_SAVE_PATH, global_step=i)
            
            
if __name__ == '__main__':
    train(datasets.Datasets(is_train=True),datasets.Datasets(is_train=False))
    
