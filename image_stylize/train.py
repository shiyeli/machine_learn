#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/7/20

import model
import tensorflow as tf
from tensorflow.contrib import slim
from PIL import Image
import matplotlib.pyplot as plt




def load_style_img(styleImgPath):
    img = tf.read_file(styleImgPath)
    style_img = tf.image.decode_jpeg(img, 3)

    style_img = tf.image.resize_images(style_img, [256, 256])

    style_img = load_data.img_process(style_img, True)  # True for substract means

    images = tf.expand_dims(style_img, 0)
    style_imgs = tf.concat([images, images, images, images], 0)  # batch is 4
    # style_imgs = tf.image.resize_images(style_imgs, [256, 256])

    return style_imgs

def train():
    style_image_path = 'images/stars.jpg'
    origin_image_path = 'images/sleep.jpg'
    generate_image_save = 'images/generate.jpg'
    
    starts_image = Image.open(style_image_path)
    sleep_image = Image.open(origin_image_path)

    # 白板图片
    black_image = np.zeros(sleep_image.shape)
    # 白板图片经过生成网络得到图片
    generate_image = inference([black_image])[0]

    # generate_image->vgg16
    generate_image_net = vgg.vgg_16(reshape_image(generate_image), sleep_image.shape)

    # content_image->vgg16
    content_image_net = vgg.vgg_16(reshape_image(sleep_image), sleep_image.shape)

    # style_image->vgg16
    style_image_net = vgg.vgg_16(reshape_image(starts_image), starts_image.shape)

    # 两张图像经过预训练好的分类网络，若提取出的高维特征之间的欧氏距离越小，则这两张图像内容越相似
    content_loss = cal_distance_tf(generate_image_net, content_image_net)

    # 风格loss:
    # style_image_net 与generate_image_net的协方差
    gram =
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        # tf.get_variable_scope().reuse_variables()
        
        sleep = get_reshaped_image('images/sleep.jpg')
        starts = get_reshaped_image('images/stars.jpg')
        
        features = vgg_inference(sess, [sleep, starts, sess.run(black_image)])
        
        sleep_feature = features[0]
        starts_feature = features[1]
        black_image_feature = features[2]
        
        style_loss = cal_distance(starts_feature, black_image_feature)
        content_loss = cal_distance(sleep_feature, black_image_feature)
        
        loss = 0.3 * style_loss + 0.7 * content_loss
        
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        
        for i in range(10000):
            _, loss = sess.run([train_op, loss])
            
            if i % 1000 == 0:
                print(i)
                black_image_ = sess.run(black_image)
                show(black_image)


if __name__ == '__main__':
    train()





