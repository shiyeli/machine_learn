#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/7/11


import vgg
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE=vgg.vgg_16.default_image_size
VGG16_CKPT='tmp/vgg_16.ckpt'
TMP_DIR='tmp'

def get_vgg_16_graph(path=VGG16_CKPT):
    """保存 vgg16 计算图 """
    x = tf.placeholder(tf.float32,[None,IMAGE_SIZE,IMAGE_SIZE,3])
    
    vgg.vgg_16(x)
    saver=tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, path)
        writer = tf.summary.FileWriter(TMP_DIR, sess.graph)
        writer.close()
        
        

        
        