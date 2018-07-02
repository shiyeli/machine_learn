#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test.py in tensorflow_1.4.0
# Created by yetongxue at 2018/7/2 20:00



# 图片转TFRecord文件

import tensorflow as tf
import glob, os
from PIL import Image
import numpy as np

BATCH_SIZE = 5
FLOWER_PHOTOS = 'flower_photos'
IS_TEST = True
OUTPUT = os.path.join(FLOWER_PHOTOS, 'output.tfrecords')
IMAGE_SIZE = 299


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_tfexample(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image),
        'label': int64_feature(label)
    }))


def generate_tfrecord(writer):
    sub_dirs = [_[0] for _ in os.walk(FLOWER_PHOTOS)][1:]

    for index, dirname in enumerate(sub_dirs):
        print 'label:%d, flower name: %s' % (index, os.path.basename(dirname))

        # 拼接glob匹配的文件名
        re = os.path.join(dirname, '*.jpg')
        files = glob.glob(re)[:10] if IS_TEST else glob.glob(re)

        for path in files:
            image = Image.open(path)
            resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            image_bytes = resized.tobytes()
            example = to_tfexample(image_bytes, index)
            writer.write(example.SerializeToString())


# 执行数据转换
def test_write():
    with tf.python_io.TFRecordWriter(OUTPUT) as writer:
        generate_tfrecord(writer)




# 读取TFRecord文件

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 解析example
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    single_image = tf.decode_raw(features['image'], tf.uint8)
    single_image = tf.reshape(single_image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    single_label = tf.cast(features['label'], tf.int32)

    return single_image, single_label


def get_batch():

    filename_queue = tf.train.string_input_producer([OUTPUT])
    single_image, single_label = read_and_decode(filename_queue)

    image_batch, label_batch = tf.train.shuffle_batch(
        [single_image, single_label],
        batch_size=BATCH_SIZE,
        num_threads=4,
        capacity=50000,
        min_after_dequeue=10000
    )
    return image_batch,label_batch


def test_read():
    image_batch, label_batch = get_batch()

    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 获取10批次数据
        for _ in range(10):
            _image_batch, _label_batch = sess.run([image_batch,label_batch])
            print _image_batch.shape,_label_batch

        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    # test_write()
    test_read()


    """
    (5, 299, 299, 3) [2 3 3 0 0]
    (5, 299, 299, 3) [0 0 2 3 2]
    (5, 299, 299, 3) [4 4 4 1 2]
    (5, 299, 299, 3) [3 4 3 4 2]
    (5, 299, 299, 3) [4 1 2 0 1]
    (5, 299, 299, 3) [2 0 0 2 3]
    (5, 299, 299, 3) [4 3 0 4 1]
    (5, 299, 299, 3) [0 4 3 2 4]
    (5, 299, 299, 3) [2 4 1 4 3]
    (5, 299, 299, 3) [2 2 1 3 0]

    """