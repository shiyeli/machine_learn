#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/7/24

import tensorflow as tf
import glob
import os
import numpy as np
from scipy.misc import imsave


def load_style_img(path):
    img = tf.read_file(path)
    style_img = tf.image.decode_jpeg(img, 3)
    style_img = tf.image.resize_images(style_img, [256, 256])
    style_img = tf.image.per_image_standardization(style_img)

    images = tf.expand_dims(style_img, 0)
    style_imgs = tf.concat([images, images, images, images], 0)

    return style_imgs


def load_test_img(path):
    img = tf.read_file(path)
    style_img = tf.image.decode_jpeg(img, 3)
    # style_img = tf.image.resize_images(style_img, [256, 256])
    style_img = tf.image.per_image_standardization(style_img)

    images = tf.expand_dims(style_img, 0)
    return images


def load_train_img(path, batch_size, scale_size, scale=False, is_gray_scale=False):
    file_glob = os.path.join(path, '*.jpg')
    file_names = glob.glob(file_glob)

    filename_quene = tf.train.string_input_producer(list(file_names), shuffle=False, seed=None)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_quene)
    image_decode = tf.image.decode_jpeg(data, channels=3)
    image_resize = tf.image.resize_images(image_decode, size=[300, 300])
    if is_gray_scale:
        image = tf.image.rgb_to_grayscale(image_resize)
    image = tf.image.per_image_standardization(image)
    quene = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=5000 + 3 * batch_size,
                                   min_after_dequeue=5000, name='synthetic_inputs')
    if scale:
        quene = tf.image.crop_to_bounding_box(quene, 0, 0, 64, 64)
        quene = tf.image.resize_nearest_neighbor(quene, [scale_size, scale_size])
    else:
        quene = tf.image.resize_nearest_neighbor(quene, [scale_size, scale_size])

    image = tf.to_float(quene)
    image = tf.cast(image, tf.float32)
    return image


def save_images(images, path):
    for i in range(images.shape[0]):
        img = images[i]

        if i == 0:
            res = img
        else:
            res = np.hstack([res, img])
    imsave(path, res)
