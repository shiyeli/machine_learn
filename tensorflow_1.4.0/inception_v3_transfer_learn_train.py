#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/28


import inception_v3_transfer_learn_datasets as datasets
import inception_v3_transfer_learn_inferance as inferance
import tensorflow as tf

LEARNING_RATE = 0.1
LEARNING_RATE_DECAY=0.5

STEPS = 1000
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

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss',loss)
        
        
        
    with tf.name_scope('train_step'):
        learing_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,STEPS,LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss, global_step=global_step)
        tf.summary.scalar('learning_rate',learing_rate)
        
        
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)

    
    # 将当前计算图输出到TensorBoard日志文件
    summary_writer = tf.summary.FileWriter('tmp/transfer/log', tf.get_default_graph())

    # 合并所有summary
    summary_merged = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            xs, ys = datasets.next_batch(BATCH)
            _summary_merged,_,_step=sess.run([summary_merged,train_step,global_step], feed_dict={x: xs, y_: ys})
            summary_writer.add_summary(_summary_merged,i)
            
            
            if i % 100 == 0:
                loss_eval ,learing_rate_= sess.run([loss,learing_rate], feed_dict={x: xs, y_: ys})
                accuracy_score = sess.run(accuracy,feed_dict={x: test_datasets.datasets[0], y_: test_datasets.datasets[1]})
                print 'Step:%d, loss=%f, accuracy=%f, learing_rate=%f' % (i, loss_eval, accuracy_score,learing_rate_)

            # 保存模型
            saver.save(sess, MODE_SAVE_PATH, global_step=i)
    
        summary_writer.close()
    
    

if __name__ == '__main__':
    train(datasets.Datasets(is_train=True), datasets.Datasets(is_train=False))

"""
Step:0, loss=5.139197, accuracy=0.323881, learing_rate=0.099931
Step:100, loss=0.107415, accuracy=0.867164, learing_rate=0.093239
Step:200, loss=0.101236, accuracy=0.891045, learing_rate=0.086995
Step:300, loss=0.047236, accuracy=0.904478, learing_rate=0.081169
Step:400, loss=0.056991, accuracy=0.901493, learing_rate=0.075733
Step:500, loss=0.081982, accuracy=0.900000, learing_rate=0.070662
Step:600, loss=0.030440, accuracy=0.908955, learing_rate=0.065930
Step:700, loss=0.026350, accuracy=0.913433, learing_rate=0.061515
Step:800, loss=0.019151, accuracy=0.911940, learing_rate=0.057395
Step:900, loss=0.017639, accuracy=0.908955, learing_rate=0.053552

Process finished with exit code 0
"""
