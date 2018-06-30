#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/28


import inception_v3_transfer_learn_datasets as datasets
import inception_v3_transfer_learn_inferance as inferance
import tensorflow as tf
import numpy as np

LEARNING_RATE=0.01
STEPS=5000
BATCH=100
MODE_SAVE_PATH='tmp/transfer/model.ckpt'
TEST_IMAGES=500


def train(datasets,test_datasets):
    
    # 输入变量初始化
    x = tf.placeholder(tf.float32, [None,inferance.POOL_3_RESHAPE_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inferance.NUM_CLASS], name='y-input')

    # 向前传播
    y = inferance.inference_custom(x)
    global_step = tf.Variable(0, trainable=False)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=y,
            labels=y_
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


"""
Step:0, loss=2.388661, accuracy=0.185075
Step:100, loss=0.436441, accuracy=0.759701
Step:200, loss=0.399566, accuracy=0.837313
Step:300, loss=0.274415, accuracy=0.849254
Step:400, loss=0.310653, accuracy=0.861194
Step:500, loss=0.288526, accuracy=0.861194
Step:600, loss=0.162615, accuracy=0.864179
Step:700, loss=0.133922, accuracy=0.874627
Step:800, loss=0.180500, accuracy=0.874627
Step:900, loss=0.155182, accuracy=0.871642
Step:1000, loss=0.107236, accuracy=0.877612
Step:1100, loss=0.152630, accuracy=0.873134
Step:1200, loss=0.088059, accuracy=0.880597
Step:1300, loss=0.098562, accuracy=0.877612
Step:1400, loss=0.077600, accuracy=0.885075
Step:1500, loss=0.066993, accuracy=0.883582
Step:1600, loss=0.070961, accuracy=0.888060
Step:1700, loss=0.110139, accuracy=0.889552
Step:1800, loss=0.118971, accuracy=0.885075
Step:1900, loss=0.099207, accuracy=0.886567
Step:2000, loss=0.083176, accuracy=0.889552
Step:2100, loss=0.049836, accuracy=0.888060
Step:2200, loss=0.066672, accuracy=0.886567
Step:2300, loss=0.055494, accuracy=0.885075
Step:2400, loss=0.070948, accuracy=0.889552
Step:2500, loss=0.061928, accuracy=0.891045
Step:2600, loss=0.037245, accuracy=0.891045
Step:2700, loss=0.108743, accuracy=0.891045
Step:2800, loss=0.042952, accuracy=0.886567
Step:2900, loss=0.049972, accuracy=0.888060
Step:3000, loss=0.053056, accuracy=0.886567
Step:3100, loss=0.044115, accuracy=0.889552
Step:3200, loss=0.047757, accuracy=0.889552
Step:3300, loss=0.045896, accuracy=0.889552
Step:3400, loss=0.039155, accuracy=0.888060
Step:3500, loss=0.026642, accuracy=0.889552
Step:3600, loss=0.046851, accuracy=0.892537
Step:3700, loss=0.030351, accuracy=0.897015
Step:3800, loss=0.025560, accuracy=0.891045
Step:3900, loss=0.033320, accuracy=0.888060
Step:4000, loss=0.043596, accuracy=0.888060
Step:4100, loss=0.032422, accuracy=0.889552
Step:4200, loss=0.025927, accuracy=0.897015
Step:4300, loss=0.047577, accuracy=0.892537
Step:4400, loss=0.028105, accuracy=0.895522
Step:4500, loss=0.053650, accuracy=0.886567
Step:4600, loss=0.038980, accuracy=0.892537
Step:4700, loss=0.028102, accuracy=0.889552
Step:4800, loss=0.024473, accuracy=0.894030
Step:4900, loss=0.019278, accuracy=0.894030
"""