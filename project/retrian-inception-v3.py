#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/4/23

#参考博客：使用现有的神经网络图像模型识别新的图像类别
#https://github.com/lxzheng/machine_learning/wiki/%E6%95%99%E7%8E%B0%E6%9C%89%E7%9A%84%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9B%BE%E5%83%8F%E6%A8%A1%E5%9E%8B%E8%AF%86%E5%88%AB%E6%96%B0%E7%9A%84%E5%9B%BE%E5%83%8F%E7%B1%BB%E5%88%AB

#下载tensorflow文件：retrain.py
#https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py

#数据集下载：
#http://www.robots.ox.ac.uk/~vgg/data/
#http://download.tensorflow.org/example_images/flower_photos.tgz



"""
重新训练命令：
python retrain.py --bottleneck_dir=logs/retrain/bottlenecks --how_many_training_steps=500 --model_dir=logs/retrain/inception --summaries_dir=logs/retrain/training_summaries/basic --output_graph=logs/retrain/retrained_graph.pb --output_labels=logs/retrain/retrained_labels.txt --image_dir=/Users/yexianyong/Downloads/flower_photos
"""


#使用重新训练的模型

import tensorflow as tf

# change this as you see fit
image_path ='/Users/yexianyong/Desktop/machine_learn/project/logs/retrain/bottlenecks'

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/Users/yexianyong/Desktop/machine_learn/project/logs/retrain/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/Users/yexianyong/Desktop/machine_learn/project/logs/retrain/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
            {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
