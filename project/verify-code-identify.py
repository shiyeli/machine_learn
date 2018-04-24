#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/4/23


from captcha.image import ImageCaptcha  # 生成验证码
import numpy as np
from PIL import Image  # 这个也可以生成验证码
import random, sys, string, os, tools

LOG_DIR = tools.makedir_logs(os.path.basename(__file__)[:-3])


######################生成验证码#########################

# 获取随机字符串
def get_random_string(length=4):
    return ''.join(
            [random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length)])


IMG_PATH = os.path.join(LOG_DIR, 'img')
tools.makedir(IMG_PATH)


# 生成验证码
def gen_captcha_image():
    image = ImageCaptcha()
    captcha_txt = get_random_string()
    image.generate(captcha_txt)
    image.write(captcha_txt, os.path.join(IMG_PATH, '{}.jpg'.format(captcha_txt)))


"""
#生成10000个验证码
num=10000
if __name__ == '__main__':
    for i in range(num):
        gen_captcha_image()
        print('Creating image %d/%d' % (i+1,num))
        
"""


######################生成tfrecord文件#########################

# 生成tfrecord文件
#https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/
#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

import tensorflow as tf
import os, random, math, sys
import numpy as np
from PIL import Image

# 验证集数量
_NUM_TEST = 500

# 随机种子
_RANDOM_SEED = 0

# 数据集路径：IMG_PATH

# tfrecord文件存放路径
TF_RECORD_DIR = tools.makedir(os.path.join(LOG_DIR, 'tfrecord'))


# 判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(dataset_dir, split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True


# 获取所有验证码
def _get_filenames_and_calsses(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : bytes_feature(image_data),
        'label0': bytes_feature(label0),
        'label1': bytes_feature(label1),
        'label2': bytes_feature(label2),
        'label3': bytes_feature(label3),
    }))


# 将图片数据转换成TFRecord格式
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']
    
    with tf.Session() as sess:
        
        # 定义tfrecord文件路径+名字
        output_filename = os.path.join(TF_RECORD_DIR, split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i, filename in enumerate(filenames):
                try:
                    print('Converted image %d/%d' % (i + 1, len(filenames)))
                    # 读取图片
                    image_data = Image.open(filename)
                    # 根据模型结构修改图片尺寸
                    image_data = image_data.resize((224, 224))
                    # 灰度化
                    image_data = np.array(image_data.convert('L'))
                    # 将图片转化成bytes
                    image_data = image_data.tobytes()
                    
                    # 获取label
                    labels_string = filename.split('/')[-1][0:4]
                    labels = []
                    for i in range(4):
                        labels.append(labels_string[i].encode('utf-8'))
                    
                    # 生成protocol数据类型
                    example = image_to_tfexample(image_data,labels[0],labels[1],labels[2],labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                
                except IOError as e:
                    print(filename,e)





if _dataset_exists(TF_RECORD_DIR):
    print('tfrecord文件已存在')
else:
    photo_filenames = _get_filenames_and_calsses(IMG_PATH)
    
    # 切分训练集和测试集
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_TEST:]
    testing_filenames = photo_filenames[:_NUM_TEST]
    
    # 数据转换
    _convert_dataset('train',training_filenames,TF_RECORD_DIR)
    _convert_dataset('test',testing_filenames,TF_RECORD_DIR)
    print('生成tfrecord文件完成')
    
    
    
##########################验证码失败主要代码##############################
    
    
