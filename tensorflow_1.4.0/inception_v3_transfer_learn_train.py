#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/28


import inception_v3_transfer_learn_datasets as datasets
import inception_v3_transfer_learn_inferance as inferance
import tensorflow as tf
import numpy as np

LEARNING_RATE=0.0001
STEPS=300
BATCH=32
MODE_SAVE_PATH='tmp/transfer/model.ckpt'
TEST_IMAGES=500

def get_batch(batch,images,labels):
    img_batch,label_batch=tf.train.shuffle_batch([images,labels],batch_size=batch,num_threads=2,capacity=1000)
    return img_batch,label_batch

def train(images,labels):
    
    # 输入变量初始化
    x = tf.placeholder(tf.float32, [None, datasets.POOL_3_RESHAPE_NODE])
    y_ = tf.placeholder(tf.float32, [None, datasets.NUM_CLASS], name='y-input')

    # 向前传播
    y = inferance.inference_custom(x)
    global_step = tf.Variable(0, trainable=False)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,
            labels=tf.argmax(y_, 1)
    )
    
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(STEPS):
            xs, ys = get_batch(BATCH,images,labels)
            _,loss, step = sess.run([train_step,loss, global_step], feed_dict={x : xs,y_: ys })
            
            if i % 20 == 0:
                print 'Step:%d, loss=%f' % (step, loss)

            # 保存模型
            saver.save(sess, MODE_SAVE_PATH, global_step=step)
            
            
if __name__ == '__main__':
    images,labels=np.load(datasets.OUTPUT_FILE)
    train(images,labels)
    