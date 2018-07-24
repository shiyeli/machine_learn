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

IMAGE_SIZE = vgg.vgg_16.default_image_size
VGG16_CKPT = 'tmp_dir/vgg_16.ckpt'
TMP_DIR = 'tmp_dir'


def get_vgg_16_graph(path=VGG16_CKPT):
    """获取 vgg16 计算图 """
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])

    vgg.vgg_16(x)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, path)
        writer = tf.summary.FileWriter(TMP_DIR, sess.graph)
        writer.close()


def arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME') as arg_sc:
            return arg_sc


def image_scale(images, scale):
    shape = images.get_shape()
    weight = shape[1].value
    height = shape[2].value
    return tf.image.resize_nearest_neighbor(images, size=(weight * scale, height * scale))


def res_module(x, outchannel, name):
    with tf.variable_scope(name_or_scope=name):
        out1 = slim.conv2d(x, outchannel, [3, 3], stride=1, scope='conv1')
        out1 = relu(out1)
        out2 = slim.conv2d(out1, outchannel, [3, 3], stride=1, scope='conv2')
        out2 = relu(out2)
        return x + out2


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def relu(x):
    return tf.nn.relu(x)


def inference(images, reuse, name, is_train=True):
    """
    生成网络向前传播
    """
    images_shape = tf.shape(images)
    images = tf.pad(images, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope(name, reuse=reuse) as vs:
        out1 = slim.conv2d(images, 32, [9, 9], padding='SAME', scope='conv1')
        out1 = relu(instance_norm(out1))

        out2 = slim.conv2d(out1, 64, [3, 3], stride=2, padding='SAME', scope='conv2')
        out2 = instance_norm(out2)

        out3 = slim.conv2d(out2, 128, [3, 3], stride=2, padding='SAME', scope='conv3')
        out3 = instance_norm(out3)

        # transform
        out4 = res_module(out3, 128, name='residual1')
        out4 = res_module(out4, 128, name='residual2')
        out4 = res_module(out4, 128, name='residual3')
        out4 = res_module(out4, 128, name='residual4')

        out5 = image_scale(out4, 2)
        out5 = slim.conv2d(out5, 64, [3, 3], stride=1, padding='SAME', scope='conv5')
        out5 = relu(instance_norm(out5))

        out6 = image_scale(out5, 2)
        out6 = slim.conv2d(out6, 32, [3, 3], stride=1, padding='SAME', scope='conv6')
        out6 = relu(instance_norm(out6))

        out = slim.conv2d(out6, 3, [9, 9], padding='SAME', scope='conv7')
        out = tf.nn.tanh(instance_norm(out))

        variables = tf.contrib.framework.get_variables(vs)
        out = (out + 1) * 127.5

        height = out.get_shape()[1].value
        width = out.get_shape()[2].value

        out = tf.image.crop_to_bounding_box(out, 10, 10, height - 20, width - 20)

    return out, variables


def styleloss(f1, f2, f3, f4):
    gen_f, _, style_f = tf.split(f1, 3, 0)
    size = tf.size(gen_f)
    style_loss = tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f2, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f3, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f4, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    return style_loss


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams
