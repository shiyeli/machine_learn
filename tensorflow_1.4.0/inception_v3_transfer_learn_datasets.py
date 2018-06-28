#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/6/28


# 获取数据

import os
import glob
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf

FLOWER_PHOTOS_PATH = 'flower_photos'
TEST_ACCOUNT = 1000

NUM_CLASS = 5
OUTPUT_FILE = 'flower_photos/flower_processed.npy'

DECODE_JPEG_CONTENTS = 'DecodeJpeg/contents:0'
POOL_3_RESHAPE_NAME = 'pool_3/_reshape:0'

# 读取inception-v3.pd
INCEPTION_V3_PD = 'tmp/inception_v3/classify_image_graph_def.pb'

IS_TEST=True

def get_datasets():
    sub_dirs = [_[0] for _ in os.walk(FLOWER_PHOTOS_PATH)][1:]
    images = []
    
    """
    flower_photos/daisy 0
    flower_photos/dandelion 1
    flower_photos/roses 2
    flower_photos/sunflowers 3
    flower_photos/tulips 4
    """
    labels = []
    
    for index, sub_dir in enumerate(sub_dirs):
        file_names = glob.glob(sub_dir + '/*.jpg')
        images.extend(file_names)
        labels.extend(np.full(len(file_names), index))
        
        if IS_TEST:
            images=images[:2]
            labels=labels[:2]
            break
        
    
    
    # 乱序
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    """
    ['flower_photos/tulips/3909355648_42cb3a5e09_n.jpg',
    'flower_photos/tulips/5634767665_0ae724774d.jpg',
    'flower_photos/dandelion/17482158576_86c5ebc2f8.jpg',
    'flower_photos/tulips/16265876844_0a149c4f76.jpg',
    'flower_photos/dandelion/344318990_7be3fb0a7d.jpg']
    """
    return images, labels


# 将数据转化成自定义全连接网络可用数据

def get_labels_one_hot(labels):
    one_hot = np.eye(NUM_CLASS)
    labels_one_hot = np.apply_along_axis(lambda x: one_hot[x], 0, labels)
    return labels_one_hot


def get_pool_3_reshape_values(sess, images):
    """
    通过inception-v3，将图片处理成pool_3_reshape数据，以供自定义全连接网络训练使用
    """
    with tf.gfile.FastGFile(INCEPTION_V3_PD, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
        decode_jpeg_contents_tensor, pool_3_reshape_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[DECODE_JPEG_CONTENTS, POOL_3_RESHAPE_NAME]
        )
    print decode_jpeg_contents_tensor,pool_3_reshape_tensor
    
    images_2048 = []
    for path in images:
        img = get_pool_3_reshape_sigal_image_values(sess, pool_3_reshape_tensor, path)
        images_2048.append(img)
    
    return images_2048


def get_pool_3_reshape_sigal_image_values(sess, pool3_reshape_tensor, image_path):
    image_raw_data = gfile.FastGFile(image_path, 'rb').read()
    #     image_data=tf.image.decode_jpeg(image_raw_data)
    pool3_reshape_value = sess.run(pool3_reshape_tensor, feed_dict={
        'import/DecodeJpeg/contents:0': image_raw_data
    })

    """
    注意获取到的tensor会默认加上import/，在feed_dict时候需要加上否则
    计算图上无法找到
    """
    return pool3_reshape_value



if __name__ == '__main__':
    images, labels = get_datasets()
    
    with tf.Session() as sess:
        images_2048 = get_pool_3_reshape_values(sess, images)
    
    labels_one_hot = get_labels_one_hot(labels)
    processed_data = np.array([np.array(images_2048), np.array(labels_one_hot)])
    # np.save(OUTPUT_FILE, processed_data)
    
    print np.array(images_2048).shape,np.array(labels_one_hot).shape