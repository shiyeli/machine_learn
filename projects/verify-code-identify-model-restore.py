#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/5/17


import os
import tensorflow as tf
from  PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt

CHAR_SET_LEN = 10
IMAGE_HEIGH = 60
IMAGE_WIDTH = 160
BATCH_SIZE = 1

TFRECORD_FILE = '/Users/yexianyong/Desktop/machine_learn/project/logs/verify-code-identify/tfrecord/test.tfrecords'
MODEL_PATH = '/Users/yexianyong/Desktop/machine_learn/project/logs/verify-code-identify/model/chptcha.model-6000'

x = tf.placeholder(tf.float32, [None, 224, 224])


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    
    # 返回文件名与文件
    _, serizlized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serizlized_example,
            features={
                'image' : tf.FixedLenFeature([], tf.string),
                'label0': tf.FixedLenFeature([], tf.int64),
                'label1': tf.FixedLenFeature([], tf.int64),
                'label2': tf.FixedLenFeature([], tf.int64),
                'label3': tf.FixedLenFeature([], tf.int64),
            })
    
    image = tf.decode_raw(features['image'], tf.uint8)
    image_raw = tf.reshape(image, [224, 224])
    image = tf.reshape(image, [224, 224])
    
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    
    return image, image_raw, label0, label1, label2, label3


image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)



# 使用shuffle_batch
image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, image_raw, label0, label1, label2, label3],
        batch_size=BATCH_SIZE,
        capacity=50000,
        min_after_dequeue=10000,
        num_threads=1
)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
        'alexnet_v2',
        num_classes=CHAR_SET_LEN,
        weight_decay=0.0005,
        is_training=False
)

with tf.Session() as sess:
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    
    logits0, logits1, logits2, logits3, endpoints = train_network_fn(X)
    
    predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])
    predict0 = tf.argmax(predict0, 1)
    
    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
    predict1 = tf.argmax(predict1, 1)
    
    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
    predict2 = tf.argmax(predict2, 1)
    
    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
    predict3 = tf.argmax(predict3, 1)
    
    # 初始化
    sess.run(tf.global_variables_initializer())
    
    # 加载训练好的模型
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_PATH)
    
    # 创建协调器管理线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for i in range(10):
        # 获取一个批次的数据标签
        b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run([image_batch,
                                                                                 image_raw_batch,
                                                                                 label_batch0,
                                                                                 label_batch1,
                                                                                 label_batch2,
                                                                                 label_batch3
                                                                                 ])

        
        print('label:', b_label0, b_label1, b_label2, b_label3)
        label0, label1, label2, label3 = sess.run([predict0, predict1, predict2, predict3], feed_dict={x: b_image})
        print('predict:', label0, label1, label2, label3)
        
        plt.imshow(b_image_raw[0])
        plt.show(block=True)
        # plt.interactive(False)
        
    # 关闭线程
    coord.request_stop()
    coord.join(threads)
