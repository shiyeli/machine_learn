#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/28


# 迁移网络结构-向前传播-自定义部分
import tensorflow as tf

POOL_3_RESHAPE_NODE = 2048
MIDDLE_NODE = 500
NUM_CLASS = 5


def inference_custom(input_tensor):
    # 输入为pool3_reshape_tensor,shape:[1,2048]

    # 自定义全连接层一
    with tf.variable_scope('custom_layer_1'):
        cl1_weights = tf.get_variable(
            'weights',
            [POOL_3_RESHAPE_NODE, MIDDLE_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )

        cl1_biases = tf.get_variable(
            'biases',
            [MIDDLE_NODE],
            initializer=tf.constant_initializer(0.1)
        )

        cl1 = tf.nn.relu(tf.matmul(input_tensor, cl1_weights) + cl1_biases)

    # 自定义全连接层二
    with tf.variable_scope('custom_layer_2'):
        cl2_weights = tf.get_variable(
            'weights',
            [MIDDLE_NODE, NUM_CLASS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )

        cl2_biases = tf.get_variable(
            'biases',
            [NUM_CLASS],
            initializer=tf.constant_initializer(0.1)
        )

        logit = tf.matmul(cl1, cl2_weights) + cl2_biases

    return logit
