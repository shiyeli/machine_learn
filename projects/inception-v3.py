#!/usr/bin/env python
# -*- coding: utf-8 -*-
# inception-v3.py in machine_learn
# Created by yetongxue at 2018/3/30 18:12

import os
import tools
import tensorflow as tf
import tarfile
import requests
import matplotlib.image as mpimage
import matplotlib.pyplot as plt

LOG_DIR = tools.makedir_logs(os.path.basename(__file__)[:-3])

#############inception-v3的下载#################

inception_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 获取文件名和文件路径
filename = inception_model_url.split('/')[-1]
filepath = os.path.join(LOG_DIR, filename)

# 下载模型
if not os.path.exists(filepath):
    print(filename, 'is downloading...')
    r = requests.get(inception_model_url)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(filename, 'is downloaded.')
else:
    print(filename, 'is already exists.')

# 解压文件
tarfile.open(filepath, 'r:gz').extractall(LOG_DIR)


################解析inception-v3中的lable######################

class NodeLookup(object):
    def __init__(self):
        label_lookup_path = os.path.join(LOG_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        uid_lookup_path = os.path.join(LOG_DIR, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n*********对应的分类名称的文件

        """
        n13134302	bulbous plant
        n13134531	bulbil, bulblet
        n13134844	cormous plant
        n13134947	fruit
        n13135692	fruitlet
        """
        label_proto_lines = tf.gfile.GFile(uid_lookup_path)
        uid_to_human = {}
        for line in label_proto_lines:
            line = line.strip('\n')
            parsed_items = line.split('\t')
            uid_to_human[parsed_items[0]] = parsed_items[1]

        # 加载target_class
        """
        entry {
          target_class: 981
          target_class_string: "n13054560"
        }
        entry {
          target_class: 329
          target_class_string: "n13133613"
        }
        """
        label_class_lines = tf.gfile.GFile(label_lookup_path)
        node_id_to_uid = {}
        for line in label_class_lines:
            if line.startswith('  target_class:'):
                # 获取分类编号
                target_class = int(line.split(':')[1])

            if line.startswith('  target_class_string:'):
                # 获取uid
                target_class_string = line.split(':')[1].split('"')[1]
                node_id_to_uid[target_class] = target_class_string

        # 将上面两个字典通过n15075141建立889与organism, being的链接
        node_id_to_name = {}
        for key, value in node_id_to_uid.items():
            try:
                name = uid_to_human[value]
                node_id_to_name[key] = name
            except KeyError:
                pass

        """
         878: 'earthstar',
         812: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
         981: 'bolete',
         329: 'ear, spike, capitulum'}
        """
        return node_id_to_name

    def id_to_description(self, node_id):
        """根据id查询描述字符串"""
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


################inception-v3的使用######################

# 读取google训练好的模型classify_image_graph_def.pb
inception_graph_def_file = os.path.join(LOG_DIR, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    # 创建一个图来存放训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # 保存图结构：保存之前将之前保存的events.out.tfevents......iMac文件删除
    tools.delete_dir_file(LOG_DIR, 'iMac')

    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    writer.close()

###############使用inception-v3识别图片###################
import numpy

VERIFY_IMAGE_PATH = '/Users/yexianyong/Downloads/images'

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    node_lookup = NodeLookup()
    # 载入图片
    for root, dirs, files in os.walk(VERIFY_IMAGE_PATH):
        for f in files:
            image_data = tf.gfile.FastGFile(os.path.join(root, f), 'rb').read()
            predixtions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predixtions = numpy.squeeze(predixtions)

            top_ks = predixtions.argsort()[-5:][::-1]
            node_id = top_ks[0]
            description_string = node_lookup.id_to_description(node_id)
            score = predixtions[node_id]
            print('Pic is {},Identified as {},and the score is:{}'.format(f, description_string, score))

#############查看测试图片################
# 下载的验证图片存放在/Users/yexianyong/Downloads/images
import math

for root, dirs, files in os.walk(VERIFY_IMAGE_PATH):
    count = len(files)
    fig, axs = plt.subplots(math.ceil(count / 4.0), 4)
    axs_1d = numpy.array(axs).flat
    for i in range(count):
        file = files[i]
        img = mpimage.imread(os.path.join(root, file))
        ax = axs_1d[i]
        ax.set_axis_off()
        ax.imshow(img)

    plt.show()

######################识别结果######################
"""
Pic is big_breast_girl.jpeg,Identified as maillot,and the score is:0.4343317151069641
Pic is cat.jpeg,Identified as Egyptian cat,and the score is:0.31411173939704895
Pic is cat_and_dog.jpeg,Identified as golden retriever,and the score is:0.8292165398597717
Pic is dog.jpeg,Identified as Pomeranian,and the score is:0.8750643730163574
Pic is egg.jpeg,Identified as balance beam, beam,and the score is:0.1599457710981369
Pic is female.jpeg,Identified as diaper, nappy, napkin,and the score is:0.26608550548553467
Pic is flower.jpeg,Identified as pot, flowerpot,and the score is:0.7364047169685364
Pic is monkey.jpeg,Identified as langur,and the score is:0.6609277725219727
Pic is police.jpeg,Identified as scuba diver,and the score is:0.31524330377578735
Pic is rabbit.jpeg,Identified as Angora, Angora rabbit,and the score is:0.6405506730079651
Pic is SUV.jpeg,Identified as pickup, pickup truck,and the score is:0.2281610667705536
Pic is tiger_lion.jpeg,Identified as tiger, Panthera tigris,and the score is:0.6159482002258301
Pic is vegetables.jpeg,Identified as strawberry,and the score is:0.2322154939174652
"""
