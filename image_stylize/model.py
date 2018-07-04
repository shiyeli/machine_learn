#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/7/13


import tensorflow as tf
from tensorflow.contrib import slim
import vgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_SIZE=vgg.vgg_16.default_image_size
MODEL_SAVE='tmp/model/model.ckpt'



def image_scale(images,scale):
    shape=images.get_shape()
    weight = shape[1].value
    height = shape[2].value
    return tf.image.resize_nearest_neighbor(images,size=(weight*scale,height*scale))
    

def res_module(x, outchannel, name):
    with tf.variable_scope(name_or_scope=name):
        out1 = slim.conv2d(x, outchannel, [3, 3], stride=1, scope='conv1')
        out1 = relu(out1)
        out2 = slim.conv2d(out1, outchannel, [3, 3], stride=1, scope='conv2')
        out2 = relu(out2)
        return x+out2

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def relu(x):
    return tf.nn.relu(x)


def inference(images,reuse,name,is_train=True):
    """
    生成网络向前传播
    """
    images_shape=tf.shape(images)
    images=tf.pad(images, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    
    with tf.variable_scope(name=name,reuse=reuse) as vs:
        out1 = slim.conv2d(images, 32, [9, 9], padding='SAME', scope='conv1')
        out1 = relu(instance_norm(out1))
        
        out2 = slim.conv2d(out1, 64, [3, 3],stride=2, padding='SAME', scope='conv2')
        out2 = instance_norm(out2)
        
        out3 = slim.conv2d(out2, 128, [3, 3],stride=2, padding='SAME', scope='conv3')
        out3 = instance_norm(out3)

        # transform
        out4 = res_module(out3, 128, name='residual1')
        out4 = res_module(out4, 128, name='residual2')
        out4 = res_module(out4, 128, name='residual3')
        out4 = res_module(out4, 128, name='residual4')
        
        
        out5 = image_scale(out4,2)
        out5 = slim.conv2d(out5, 64, [3, 3],stride=1, padding='SAME', scope='conv5')
        out5 = relu(instance_norm(out5))
        
        out6 = image_scale(out5, 2)
        out6 = slim.conv2d(out6, 32, [3, 3], stride=1, padding='SAME', scope='conv6')
        out6 = relu(instance_norm(out6))

        out = slim.conv2d(out6, 3, [9, 9], padding='SAME', scope='conv6')
        out = tf.nn.tanh(instance_norm(out))

        variables = tf.contrib.framework.get_variables(vs)
        out = (out + 1) * 127.5

        height = out.get_shape()[1].value
        width = out.get_shape()[2].value

        out = tf.image.crop_to_bounding_box(out, 10, 10, height - 20, width - 20)
        
    return out, variables


def vgg_inference(sess, images):
    """
    :image (None,224,224,3)
    :return   None ( 1000,out1,out2,out3,out4,out5)
    """
    
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    net= vgg.vgg_16(x)
    
    sess.run(tf.global_variables_initializer())
    net_,  = sess.run([net], feed_dict={x: images})
    return net_



#
# def cal_distance(v1, v2):
#     """计算两个向量欧式距离"""
#     return np.sqrt((np.sum(np.square(v1 - v2))))
#
# def cal_distance_tf(v1,v2):
#     return tf.sqrt((tf.reduce_sum(tf.square(v1-v2))))
#
# def show(img):
#     plt.imshow(img)
#     plt.show()




