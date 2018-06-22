#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/22


import tensorflow as tf

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接节点个数
FC_SIZE = 512


# 向前传播
def inference(input_tensor, train, regularizer):
    # 第一层卷积
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
                'weights',
                [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
                'biases',
                [CONV1_DEEP],
                initializer=tf.constant_initializer(0.0)
        )
        
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    
    # 第二层池化
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # 第三层卷积
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
                'weights',
                [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        
        conv2_biases = tf.get_variable(
                'biases',
                [CONV2_DEEP],
                initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    # 第四层池化
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        
        # 将pool2中的图片展开，注意第一维是batch,pool2_shape[0]为一批次中数据的个数

    """
    pool2_shape=pool2.get_shape().as_list()
    
    TypeError: Failed to convert object of type <type 'list'> to Tensor.
    Contents: [None, 3136].
    Consider casting elements to a supported type.
    """
    # pool2_shape = pool2.get_shape().as_list()
    pool2_shape = tf.shape(pool2)
    nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    reshaped = tf.reshape(pool2, [pool2_shape[0], nodes])
    
    
    # 第五层（全连接第一层）
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
                'weights',
                [nodes, FC_SIZE],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        
        # 只有全连接的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        
        fc1_biases = tf.get_variable(
                'biases',
                [FC_SIZE],
                initializer=tf.constant_initializer(0.1)
        )
        
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    
    # 第六层（全连接第二层）
    
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
                'weights',
                [FC_SIZE, NUM_LABELS],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        
        # 只有全连接的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        
        fc2_biases = tf.get_variable(
                'biases',
                [NUM_LABELS],
                initializer=tf.constant_initializer(0.1)
        )
        
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    
    return logit
