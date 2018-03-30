#!/usr/bin/env python
# -*- coding: utf-8 -*-
# inception-v3.py in machine_learn
# Created by yetongxue at 2018/3/30 18:12




import os
import tensorflow as tf
import tarfile
import requests

inception_model_url='http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


#模型存放地址
MODEL_DIR=os.path.join(os.getcwd(),'inception_model')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)





# 获取文件名和文件路径
filename = inception_model_url.split('/')[-1]
filepath = os.path.join(MODEL_DIR, filename)

# 下载模型
if not os.path.exists(filepath):
    print(filename, 'is downloading...')
    r = requests.get(inception_model_url)
    with open(filepath,'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(filename,'is downloaded.')
else:
    print(filename,'is already exists.')

#解压文件
tarfile.open(filepath,'r:gz').extractall(MODEL_DIR)

#创建存放模型结构的文件夹
log_dir='inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


#读取google训练好的模型classify_image_graph_def.pb
inception_graph_def_file=os.path.join(MODEL_DIR,'classify_image_graph_def.pb')
with tf.Session() as sess:
    #创建一个图来存放训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name='')

    #保存图结构
    writer=tf.summary.FileWriter(log_dir,sess.graph)
    writer.close()

