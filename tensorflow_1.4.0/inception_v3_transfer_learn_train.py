#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/28


import inception_v3_transfer_learn_datasets as datasets
import inception_v3_transfer_learn_inferance as inferance
import tensorflow as tf

LEARNING_RATE = 0.01
LEARNING_RATE_DECAY=0.9

STEPS = 3000
BATCH = 100
MODE_SAVE_PATH = 'tmp/transfer/model.ckpt'
TEST_IMAGES = 500


def train(datasets, test_datasets):
    # 输入变量初始化
    x = tf.placeholder(tf.float32, [None, inferance.POOL_3_RESHAPE_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inferance.NUM_CLASS], name='y-input')

    # 向前传播
    y = inferance.inference_custom(x)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=y,
        labels=y_
    )

    loss = tf.reduce_mean(cross_entropy)
    
    learing_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,STEPS,LEARNING_RATE_DECAY)
    
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            xs, ys = datasets.next_batch(BATCH)
            sess.run([train_step,global_step], feed_dict={x: xs, y_: ys})

            if i % 200 == 0:
                loss_eval = sess.run(loss, feed_dict={x: xs, y_: ys})
                accuracy_score = sess.run(accuracy,
                                          feed_dict={x: test_datasets.datasets[0], y_: test_datasets.datasets[1]})
                print 'Step:%d, loss=%f, accuracy=%f' % (i, loss_eval, accuracy_score)

            # 保存模型
            saver.save(sess, MODE_SAVE_PATH, global_step=i)


if __name__ == '__main__':
    train(datasets.Datasets(is_train=True), datasets.Datasets(is_train=False))

"""
Step:0, loss=2.685330, accuracy=0.195522
Step:200, loss=0.477912, accuracy=0.825373
Step:400, loss=0.318471, accuracy=0.858209
Step:600, loss=0.155637, accuracy=0.874627
Step:800, loss=0.179882, accuracy=0.882090
Step:1000, loss=0.102939, accuracy=0.883582
Step:1200, loss=0.096046, accuracy=0.895522
Step:1400, loss=0.105443, accuracy=0.895522
Step:1600, loss=0.050719, accuracy=0.902985
Step:1800, loss=0.078408, accuracy=0.901493
Step:2000, loss=0.083749, accuracy=0.904478
Step:2200, loss=0.053214, accuracy=0.905970
Step:2400, loss=0.050154, accuracy=0.908955
Step:2600, loss=0.090683, accuracy=0.901493
Step:2800, loss=0.044439, accuracy=0.910448

Process finished with exit code 0
"""
