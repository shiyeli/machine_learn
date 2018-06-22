#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/4/23


from captcha.image import ImageCaptcha  # 生成验证码
import numpy as np
from PIL import Image  # 这个也可以生成验证码
import random, string, os, tools

LOG_DIR = tools.makedir_logs(os.path.basename(__file__)[:-3])


######################生成验证码#########################

# 获取随机字符串
def get_random_string(length=4):
    return ''.join(
            [random.choice(string.digits) for _ in range(length)])


created,IMG_PATH =tools.makedir(os.path.join(LOG_DIR, 'images'))


# 生成验证码
def gen_captcha_image():
    image = ImageCaptcha()
    captcha_txt = get_random_string()
    image.generate(captcha_txt)
    image.write(captcha_txt, os.path.join(IMG_PATH, '{}.jpg'.format(captcha_txt)))


#生成10000个验证码
num=10000
if created:
    for i in range(num):
        gen_captcha_image()
        print('Creating image %d/%d' % (i+1,num))


######################生成tfrecord文件#########################

# 生成tfrecord文件
#https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/
#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

import tensorflow as tf
import os, random
import numpy as np
from PIL import Image

# 验证集数量
_NUM_TEST = 500

# 随机种子
_RANDOM_SEED = 0

# 数据集路径：IMG_PATH

# tfrecord文件存放路径
created,TF_RECORD_DIR = tools.makedir(os.path.join(LOG_DIR, 'tfrecord'))

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

def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : bytes_feature(image_data),
        'label0': int64_feature(label0),
        'label1': int64_feature(label1),
        'label2': int64_feature(label2),
        'label3': int64_feature(label3),
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
                        labels.append(int(labels_string[i]))
                    
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

"""
验证码识别方法一
将label转化成一维向量进行分类，与手写数字识别类似


识别方法二
Multi-task Learning 交替训练



"""
    
from nets import nets_factory

# 不同字符数量
CHAR_SET_LEN = 10
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 25
# tfrecord文件存放路径
TFRECORD_FILE = os.path.join(LOG_DIR,'tfrecord/train.tfrecords')

MODEL_SAVE_PATH=os.path.join(LOG_DIR,'model/chptcha.model')

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])

# 学习率
lr = tf.Variable(0.003, dtype=tf.float32)


# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
            features={
                'image' : tf.FixedLenFeature([], tf.string),
                'label0': tf.FixedLenFeature([], tf.int64),
                'label1': tf.FixedLenFeature([], tf.int64),
                'label2': tf.FixedLenFeature([], tf.int64),
                'label3': tf.FixedLenFeature([], tf.int64),
            })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    
    return image, label0, label1, label2, label3




# 获取图片数据和标签
image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

# 使用shuffle_batch可以随机打乱
image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, label0, label1, label2, label3], batch_size=BATCH_SIZE,
        capacity=50000, min_after_dequeue=10000, num_threads=1)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
        'alexnet_v2',
        num_classes=CHAR_SET_LEN,
        weight_decay=0.0005,
        is_training=True)

with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)
    
    # 把标签转成one_hot的形式
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)
    
    # 计算loss
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits0, labels=one_hot_labels0))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1, labels=one_hot_labels1))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=one_hot_labels2))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits3, labels=one_hot_labels3))
    # 计算总的loss
    total_loss = (loss0 + loss1 + loss2 + loss3) / 4.0
    # 优化total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
    
    # 计算准确率
    correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0, 1), tf.argmax(logits0, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))
    
    correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1, 1), tf.argmax(logits1, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    
    correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2, 1), tf.argmax(logits2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
    
    correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3, 1), tf.argmax(logits3, 1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
    
    # 用于保存模型
    saver = tf.train.Saver()
    # 初始化
    sess.run(tf.global_variables_initializer())
    
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for i in range(6001):
        # 获取一个批次的数据和标签
        b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
                [image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
        # 优化模型
        sess.run(optimizer, feed_dict={x: b_image, y0: b_label0, y1: b_label1, y2: b_label2, y3: b_label3})
        
        # 每迭代20次计算一次loss和准确率
        if i % 20 == 0:
            # 每迭代2000次降低一次学习率
            if i % 2000 == 0:
                sess.run(tf.assign(lr, lr / 3))
            acc0, acc1, acc2, acc3, loss_ = sess.run([accuracy0, accuracy1, accuracy2, accuracy3, total_loss],
                    feed_dict={x : b_image,
                               y0: b_label0,
                               y1: b_label1,
                               y2: b_label2,
                               y3: b_label3})
            learning_rate = sess.run(lr)
            print("Iter:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.4f" % (
            i, loss_, acc0, acc1, acc2, acc3, learning_rate))
            
            # 保存模型
            # if acc0 > 0.90 and acc1 > 0.90 and acc2 > 0.90 and acc3 > 0.90:
            if i == 6000:
                saver.save(sess, MODEL_SAVE_PATH, global_step=i)
                break
                
                # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
    
    
    
"""
tfrecord文件已存在
Iter:0  Loss:1762.182  Accuracy:0.24,0.12,0.20,0.20  Learning_rate:0.0010
Iter:20  Loss:2.283  Accuracy:0.12,0.16,0.28,0.20  Learning_rate:0.0010
Iter:40  Loss:2.297  Accuracy:0.16,0.24,0.04,0.16  Learning_rate:0.0010
Iter:60  Loss:2.307  Accuracy:0.12,0.12,0.12,0.00  Learning_rate:0.0010
Iter:80  Loss:2.309  Accuracy:0.04,0.12,0.04,0.12  Learning_rate:0.0010
Iter:100  Loss:2.299  Accuracy:0.16,0.04,0.04,0.12  Learning_rate:0.0010
Iter:120  Loss:2.305  Accuracy:0.08,0.08,0.12,0.12  Learning_rate:0.0010
Iter:140  Loss:2.300  Accuracy:0.12,0.12,0.08,0.16  Learning_rate:0.0010
Iter:160  Loss:2.303  Accuracy:0.08,0.04,0.08,0.16  Learning_rate:0.0010
Iter:180  Loss:2.303  Accuracy:0.08,0.04,0.12,0.08  Learning_rate:0.0010
Iter:200  Loss:2.309  Accuracy:0.12,0.04,0.20,0.12  Learning_rate:0.0010
Iter:220  Loss:2.318  Accuracy:0.08,0.12,0.16,0.00  Learning_rate:0.0010
Iter:240  Loss:2.310  Accuracy:0.08,0.04,0.08,0.16  Learning_rate:0.0010
Iter:260  Loss:2.309  Accuracy:0.04,0.16,0.12,0.08  Learning_rate:0.0010
Iter:280  Loss:2.288  Accuracy:0.20,0.12,0.16,0.04  Learning_rate:0.0010
Iter:300  Loss:2.329  Accuracy:0.00,0.08,0.12,0.04  Learning_rate:0.0010
Iter:320  Loss:2.288  Accuracy:0.16,0.00,0.16,0.16  Learning_rate:0.0010
Iter:340  Loss:2.289  Accuracy:0.16,0.12,0.16,0.12  Learning_rate:0.0010
Iter:360  Loss:2.296  Accuracy:0.00,0.12,0.08,0.12  Learning_rate:0.0010
Iter:380  Loss:2.287  Accuracy:0.20,0.04,0.08,0.08  Learning_rate:0.0010
Iter:400  Loss:2.302  Accuracy:0.12,0.04,0.08,0.08  Learning_rate:0.0010
Iter:420  Loss:2.317  Accuracy:0.00,0.04,0.08,0.04  Learning_rate:0.0010
Iter:440  Loss:2.294  Accuracy:0.12,0.12,0.16,0.08  Learning_rate:0.0010
Iter:460  Loss:2.308  Accuracy:0.12,0.04,0.20,0.04  Learning_rate:0.0010
Iter:480  Loss:2.305  Accuracy:0.12,0.04,0.04,0.08  Learning_rate:0.0010
Iter:500  Loss:2.221  Accuracy:0.24,0.08,0.12,0.12  Learning_rate:0.0010
Iter:520  Loss:2.231  Accuracy:0.28,0.12,0.12,0.08  Learning_rate:0.0010
Iter:540  Loss:2.151  Accuracy:0.52,0.04,0.08,0.00  Learning_rate:0.0010
Iter:560  Loss:2.172  Accuracy:0.44,0.04,0.12,0.12  Learning_rate:0.0010
Iter:580  Loss:2.169  Accuracy:0.32,0.04,0.08,0.04  Learning_rate:0.0010
Iter:600  Loss:2.074  Accuracy:0.44,0.16,0.16,0.12  Learning_rate:0.0010
Iter:620  Loss:1.976  Accuracy:0.60,0.28,0.04,0.20  Learning_rate:0.0010
Iter:640  Loss:2.058  Accuracy:0.60,0.20,0.16,0.08  Learning_rate:0.0010
Iter:660  Loss:1.946  Accuracy:0.64,0.08,0.08,0.16  Learning_rate:0.0010
Iter:680  Loss:2.032  Accuracy:0.68,0.08,0.08,0.08  Learning_rate:0.0010
Iter:700  Loss:1.967  Accuracy:0.60,0.16,0.08,0.04  Learning_rate:0.0010
Iter:720  Loss:1.988  Accuracy:0.60,0.16,0.12,0.04  Learning_rate:0.0010
Iter:740  Loss:2.001  Accuracy:0.76,0.04,0.08,0.12  Learning_rate:0.0010
Iter:760  Loss:1.930  Accuracy:0.68,0.08,0.12,0.08  Learning_rate:0.0010
Iter:780  Loss:2.028  Accuracy:0.52,0.12,0.08,0.08  Learning_rate:0.0010
Iter:800  Loss:1.883  Accuracy:0.84,0.08,0.04,0.08  Learning_rate:0.0010
Iter:820  Loss:1.880  Accuracy:0.64,0.16,0.20,0.16  Learning_rate:0.0010
Iter:840  Loss:1.894  Accuracy:0.72,0.08,0.24,0.08  Learning_rate:0.0010
Iter:860  Loss:1.798  Accuracy:0.88,0.16,0.20,0.04  Learning_rate:0.0010
Iter:880  Loss:1.895  Accuracy:0.84,0.08,0.12,0.08  Learning_rate:0.0010
Iter:900  Loss:1.807  Accuracy:0.88,0.08,0.12,0.16  Learning_rate:0.0010
Iter:920  Loss:1.845  Accuracy:0.88,0.16,0.08,0.16  Learning_rate:0.0010
Iter:940  Loss:1.815  Accuracy:0.96,0.12,0.16,0.08  Learning_rate:0.0010
Iter:960  Loss:1.762  Accuracy:1.00,0.20,0.04,0.16  Learning_rate:0.0010
Iter:980  Loss:1.800  Accuracy:0.84,0.24,0.16,0.20  Learning_rate:0.0010
Iter:1000  Loss:1.764  Accuracy:0.96,0.12,0.16,0.16  Learning_rate:0.0010
Iter:1020  Loss:1.818  Accuracy:0.88,0.04,0.20,0.16  Learning_rate:0.0010
Iter:1040  Loss:1.684  Accuracy:0.96,0.16,0.12,0.24  Learning_rate:0.0010
Iter:1060  Loss:1.613  Accuracy:0.92,0.24,0.16,0.20  Learning_rate:0.0010
Iter:1080  Loss:1.552  Accuracy:0.84,0.12,0.20,0.48  Learning_rate:0.0010
Iter:1100  Loss:1.654  Accuracy:0.72,0.20,0.28,0.40  Learning_rate:0.0010
Iter:1120  Loss:1.365  Accuracy:0.92,0.40,0.32,0.48  Learning_rate:0.0010
Iter:1140  Loss:1.284  Accuracy:0.96,0.44,0.28,0.52  Learning_rate:0.0010
Iter:1160  Loss:1.247  Accuracy:0.92,0.36,0.36,0.48  Learning_rate:0.0010
Iter:1180  Loss:1.261  Accuracy:0.88,0.52,0.24,0.72  Learning_rate:0.0010
Iter:1200  Loss:1.145  Accuracy:1.00,0.52,0.44,0.52  Learning_rate:0.0010
Iter:1220  Loss:1.162  Accuracy:0.92,0.32,0.52,0.56  Learning_rate:0.0010
Iter:1240  Loss:1.188  Accuracy:0.96,0.28,0.56,0.56  Learning_rate:0.0010
Iter:1260  Loss:0.994  Accuracy:0.80,0.72,0.56,0.56  Learning_rate:0.0010
Iter:1280  Loss:0.989  Accuracy:0.80,0.72,0.48,0.56  Learning_rate:0.0010
Iter:1300  Loss:0.909  Accuracy:0.88,0.76,0.40,0.68  Learning_rate:0.0010
Iter:1320  Loss:0.865  Accuracy:0.84,0.72,0.68,0.64  Learning_rate:0.0010
Iter:1340  Loss:0.799  Accuracy:0.96,0.68,0.56,0.80  Learning_rate:0.0010
Iter:1360  Loss:0.750  Accuracy:0.88,0.72,0.72,0.68  Learning_rate:0.0010
Iter:1380  Loss:0.763  Accuracy:0.88,0.72,0.52,0.72  Learning_rate:0.0010
Iter:1400  Loss:0.693  Accuracy:0.96,0.68,0.64,0.80  Learning_rate:0.0010
Iter:1420  Loss:0.679  Accuracy:0.92,0.72,0.68,0.68  Learning_rate:0.0010
Iter:1440  Loss:0.616  Accuracy:0.76,0.72,0.80,0.96  Learning_rate:0.0010
Iter:1460  Loss:0.704  Accuracy:0.84,0.64,0.76,0.76  Learning_rate:0.0010
Iter:1480  Loss:0.498  Accuracy:0.96,0.72,0.84,0.88  Learning_rate:0.0010
Iter:1500  Loss:0.584  Accuracy:0.96,0.80,0.76,0.72  Learning_rate:0.0010
Iter:1520  Loss:0.610  Accuracy:0.92,0.76,0.72,0.80  Learning_rate:0.0010
Iter:1540  Loss:0.578  Accuracy:0.96,0.72,0.72,0.76  Learning_rate:0.0010
Iter:1560  Loss:0.533  Accuracy:0.92,0.80,0.80,0.88  Learning_rate:0.0010
Iter:1580  Loss:0.666  Accuracy:0.84,0.80,0.64,0.80  Learning_rate:0.0010
Iter:1600  Loss:0.538  Accuracy:1.00,0.80,0.68,0.64  Learning_rate:0.0010
Iter:1620  Loss:0.292  Accuracy:0.96,0.88,0.96,0.92  Learning_rate:0.0010
Iter:1640  Loss:0.399  Accuracy:0.96,0.88,0.80,0.84  Learning_rate:0.0010
Iter:1660  Loss:0.347  Accuracy:0.92,0.84,0.72,0.88  Learning_rate:0.0010
Iter:1680  Loss:0.448  Accuracy:0.92,0.84,0.72,0.96  Learning_rate:0.0010
Iter:1700  Loss:0.277  Accuracy:1.00,0.88,0.88,0.92  Learning_rate:0.0010
Iter:1720  Loss:0.292  Accuracy:0.96,0.84,0.88,0.92  Learning_rate:0.0010
Iter:1740  Loss:0.469  Accuracy:0.84,0.76,0.84,0.92  Learning_rate:0.0010
Iter:1760  Loss:0.211  Accuracy:1.00,0.96,0.76,0.92  Learning_rate:0.0010
Iter:1780  Loss:0.261  Accuracy:0.96,0.92,0.84,0.84  Learning_rate:0.0010
Iter:1800  Loss:0.468  Accuracy:0.96,0.80,0.72,0.92  Learning_rate:0.0010
Iter:1820  Loss:0.358  Accuracy:0.96,0.92,0.84,0.84  Learning_rate:0.0010
Iter:1840  Loss:0.380  Accuracy:0.96,0.88,0.80,0.84  Learning_rate:0.0010
Iter:1860  Loss:0.329  Accuracy:0.96,0.88,0.88,0.80  Learning_rate:0.0010
Iter:1880  Loss:0.251  Accuracy:1.00,0.92,0.88,0.84  Learning_rate:0.0010
Iter:1900  Loss:0.273  Accuracy:0.96,0.96,0.72,0.88  Learning_rate:0.0010
Iter:1920  Loss:0.210  Accuracy:0.96,0.96,0.92,0.92  Learning_rate:0.0010
Iter:1940  Loss:0.469  Accuracy:0.92,0.80,0.80,0.80  Learning_rate:0.0010
Iter:1960  Loss:0.403  Accuracy:0.92,0.88,0.84,0.88  Learning_rate:0.0010
Iter:1980  Loss:0.414  Accuracy:1.00,0.80,0.72,0.96  Learning_rate:0.0010
Iter:2000  Loss:0.212  Accuracy:0.92,0.92,0.96,0.92  Learning_rate:0.0003
Iter:2020  Loss:0.211  Accuracy:0.96,0.84,0.84,0.96  Learning_rate:0.0003
Iter:2040  Loss:0.217  Accuracy:1.00,0.88,0.88,0.96  Learning_rate:0.0003
Iter:2060  Loss:0.206  Accuracy:1.00,0.84,0.96,0.92  Learning_rate:0.0003
Iter:2080  Loss:0.236  Accuracy:0.96,0.96,0.84,0.96  Learning_rate:0.0003
Iter:2100  Loss:0.133  Accuracy:0.96,1.00,0.92,0.96  Learning_rate:0.0003
Iter:2120  Loss:0.322  Accuracy:0.88,0.96,0.96,0.88  Learning_rate:0.0003
Iter:2140  Loss:0.233  Accuracy:0.96,0.96,0.80,0.96  Learning_rate:0.0003
Iter:2160  Loss:0.123  Accuracy:1.00,0.96,0.92,0.92  Learning_rate:0.0003
Iter:2180  Loss:0.061  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0003
Iter:2200  Loss:0.233  Accuracy:1.00,0.96,0.92,0.92  Learning_rate:0.0003
Iter:2220  Loss:0.134  Accuracy:1.00,0.96,0.88,0.96  Learning_rate:0.0003
Iter:2240  Loss:0.167  Accuracy:1.00,0.92,1.00,0.88  Learning_rate:0.0003
Iter:2260  Loss:0.247  Accuracy:1.00,0.92,0.96,0.92  Learning_rate:0.0003
Iter:2280  Loss:0.200  Accuracy:0.96,1.00,0.92,0.76  Learning_rate:0.0003
Iter:2300  Loss:0.114  Accuracy:1.00,1.00,0.92,0.92  Learning_rate:0.0003
Iter:2320  Loss:0.220  Accuracy:0.92,0.96,0.96,0.92  Learning_rate:0.0003
Iter:2340  Loss:0.095  Accuracy:1.00,0.96,1.00,0.92  Learning_rate:0.0003
Iter:2360  Loss:0.197  Accuracy:1.00,0.84,0.96,0.92  Learning_rate:0.0003
Iter:2380  Loss:0.097  Accuracy:1.00,0.96,0.92,1.00  Learning_rate:0.0003
Iter:2400  Loss:0.210  Accuracy:0.96,0.88,0.84,0.92  Learning_rate:0.0003
Iter:2420  Loss:0.072  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0003
Iter:2440  Loss:0.093  Accuracy:1.00,0.96,0.92,0.96  Learning_rate:0.0003
Iter:2460  Loss:0.144  Accuracy:1.00,0.88,1.00,0.92  Learning_rate:0.0003
Iter:2480  Loss:0.071  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0003
Iter:2500  Loss:0.138  Accuracy:1.00,1.00,0.96,0.92  Learning_rate:0.0003
Iter:2520  Loss:0.145  Accuracy:1.00,0.92,0.96,0.96  Learning_rate:0.0003
Iter:2540  Loss:0.104  Accuracy:1.00,0.96,1.00,0.92  Learning_rate:0.0003
Iter:2560  Loss:0.129  Accuracy:1.00,0.92,0.96,0.96  Learning_rate:0.0003
Iter:2580  Loss:0.182  Accuracy:1.00,0.88,0.92,0.96  Learning_rate:0.0003
Iter:2600  Loss:0.158  Accuracy:1.00,0.88,0.96,1.00  Learning_rate:0.0003
Iter:2620  Loss:0.132  Accuracy:1.00,0.84,0.92,1.00  Learning_rate:0.0003
Iter:2640  Loss:0.082  Accuracy:0.96,0.96,1.00,0.92  Learning_rate:0.0003
Iter:2660  Loss:0.159  Accuracy:1.00,0.96,0.92,0.92  Learning_rate:0.0003
Iter:2680  Loss:0.058  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0003
Iter:2700  Loss:0.065  Accuracy:1.00,0.96,0.96,0.96  Learning_rate:0.0003
Iter:2720  Loss:0.186  Accuracy:1.00,0.92,0.96,0.96  Learning_rate:0.0003
Iter:2740  Loss:0.104  Accuracy:0.96,0.96,1.00,0.96  Learning_rate:0.0003
Iter:2760  Loss:0.062  Accuracy:1.00,0.96,1.00,0.96  Learning_rate:0.0003
Iter:2780  Loss:0.068  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0003
Iter:2800  Loss:0.097  Accuracy:0.96,1.00,0.92,1.00  Learning_rate:0.0003
Iter:2820  Loss:0.209  Accuracy:1.00,0.92,0.92,0.88  Learning_rate:0.0003
Iter:2840  Loss:0.063  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0003
Iter:2860  Loss:0.052  Accuracy:1.00,1.00,0.96,0.96  Learning_rate:0.0003
Iter:2880  Loss:0.118  Accuracy:1.00,0.88,1.00,1.00  Learning_rate:0.0003
Iter:2900  Loss:0.037  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0003
Iter:2920  Loss:0.072  Accuracy:1.00,0.96,0.92,0.96  Learning_rate:0.0003
Iter:2940  Loss:0.086  Accuracy:0.96,1.00,0.96,0.96  Learning_rate:0.0003
Iter:2960  Loss:0.128  Accuracy:0.92,1.00,0.92,0.92  Learning_rate:0.0003
Iter:2980  Loss:0.058  Accuracy:1.00,0.92,0.96,1.00  Learning_rate:0.0003
Iter:3000  Loss:0.148  Accuracy:1.00,1.00,0.88,0.92  Learning_rate:0.0003
Iter:3020  Loss:0.018  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0003
Iter:3040  Loss:0.166  Accuracy:0.96,0.80,1.00,0.96  Learning_rate:0.0003
Iter:3060  Loss:0.044  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0003
Iter:3080  Loss:0.030  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0003
Iter:3100  Loss:0.081  Accuracy:0.96,1.00,0.96,0.96  Learning_rate:0.0003
Iter:3120  Loss:0.060  Accuracy:1.00,1.00,0.92,0.96  Learning_rate:0.0003
Iter:3140  Loss:0.104  Accuracy:1.00,0.96,0.84,1.00  Learning_rate:0.0003
Iter:3160  Loss:0.028  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0003
Iter:3180  Loss:0.027  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0003
Iter:3200  Loss:0.088  Accuracy:0.96,0.96,1.00,1.00  Learning_rate:0.0003
Iter:3220  Loss:0.089  Accuracy:1.00,0.96,0.92,0.96  Learning_rate:0.0003
Iter:3240  Loss:0.081  Accuracy:1.00,0.88,0.96,1.00  Learning_rate:0.0003
Iter:3260  Loss:0.028  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0003
Iter:3280  Loss:0.037  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0003
Iter:3300  Loss:0.045  Accuracy:1.00,1.00,0.92,1.00  Learning_rate:0.0003
Iter:3320  Loss:0.109  Accuracy:0.96,1.00,0.92,0.92  Learning_rate:0.0003
Iter:3340  Loss:0.137  Accuracy:1.00,1.00,0.92,0.96  Learning_rate:0.0003
Iter:3360  Loss:0.093  Accuracy:0.96,1.00,0.92,0.96  Learning_rate:0.0003
Iter:3380  Loss:0.102  Accuracy:1.00,1.00,0.92,0.96  Learning_rate:0.0003
Iter:3400  Loss:0.045  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0003
Iter:3420  Loss:0.143  Accuracy:0.96,0.92,0.96,1.00  Learning_rate:0.0003
Iter:3440  Loss:0.076  Accuracy:0.96,0.96,0.96,0.96  Learning_rate:0.0003
Iter:3460  Loss:0.057  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0003
Iter:3480  Loss:0.066  Accuracy:1.00,1.00,0.96,0.96  Learning_rate:0.0003
Iter:3500  Loss:0.040  Accuracy:0.96,0.96,1.00,1.00  Learning_rate:0.0003
Iter:3520  Loss:0.029  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0003
Iter:3540  Loss:0.019  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0003
Iter:3560  Loss:0.118  Accuracy:0.96,0.96,1.00,0.84  Learning_rate:0.0003
Iter:3580  Loss:0.096  Accuracy:1.00,0.96,0.96,0.96  Learning_rate:0.0003
Iter:3600  Loss:0.044  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0003
Iter:3620  Loss:0.068  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0003
Iter:3640  Loss:0.063  Accuracy:1.00,0.96,1.00,0.92  Learning_rate:0.0003
Iter:3660  Loss:0.055  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0003
Iter:3680  Loss:0.068  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0003
Iter:3700  Loss:0.080  Accuracy:1.00,1.00,0.96,0.88  Learning_rate:0.0003
Iter:3720  Loss:0.025  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0003
Iter:3740  Loss:0.047  Accuracy:0.96,0.96,1.00,1.00  Learning_rate:0.0003
Iter:3760  Loss:0.067  Accuracy:1.00,1.00,0.96,0.96  Learning_rate:0.0003
Iter:3780  Loss:0.130  Accuracy:1.00,0.96,0.88,0.92  Learning_rate:0.0003
Iter:3800  Loss:0.032  Accuracy:0.96,1.00,1.00,1.00  Learning_rate:0.0003
Iter:3820  Loss:0.083  Accuracy:1.00,1.00,0.92,0.92  Learning_rate:0.0003
Iter:3840  Loss:0.086  Accuracy:1.00,0.96,0.96,0.96  Learning_rate:0.0003
Iter:3860  Loss:0.026  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0003
Iter:3880  Loss:0.124  Accuracy:1.00,0.96,0.92,0.96  Learning_rate:0.0003
Iter:3900  Loss:0.060  Accuracy:1.00,1.00,0.96,0.92  Learning_rate:0.0003
Iter:3920  Loss:0.073  Accuracy:0.96,0.96,0.96,1.00  Learning_rate:0.0003
Iter:3940  Loss:0.038  Accuracy:1.00,1.00,0.96,0.96  Learning_rate:0.0003
Iter:3960  Loss:0.047  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0003
Iter:3980  Loss:0.019  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0003
Iter:4000  Loss:0.032  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:4020  Loss:0.037  Accuracy:1.00,0.96,1.00,0.96  Learning_rate:0.0001
Iter:4040  Loss:0.072  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4060  Loss:0.034  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:4080  Loss:0.053  Accuracy:0.96,1.00,1.00,0.96  Learning_rate:0.0001
Iter:4100  Loss:0.020  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4120  Loss:0.036  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:4140  Loss:0.053  Accuracy:0.96,0.96,0.96,0.96  Learning_rate:0.0001
Iter:4160  Loss:0.055  Accuracy:1.00,1.00,0.96,0.96  Learning_rate:0.0001
Iter:4180  Loss:0.084  Accuracy:1.00,0.96,0.96,0.92  Learning_rate:0.0001
Iter:4200  Loss:0.047  Accuracy:1.00,0.96,1.00,0.92  Learning_rate:0.0001
Iter:4220  Loss:0.012  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4240  Loss:0.021  Accuracy:0.96,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4260  Loss:0.130  Accuracy:0.92,1.00,0.84,1.00  Learning_rate:0.0001
Iter:4280  Loss:0.025  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4300  Loss:0.044  Accuracy:1.00,1.00,0.88,1.00  Learning_rate:0.0001
Iter:4320  Loss:0.027  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4340  Loss:0.076  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4360  Loss:0.007  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4380  Loss:0.006  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4400  Loss:0.013  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4420  Loss:0.013  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4440  Loss:0.043  Accuracy:1.00,1.00,0.96,0.96  Learning_rate:0.0001
Iter:4460  Loss:0.007  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4480  Loss:0.064  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:4500  Loss:0.012  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4520  Loss:0.125  Accuracy:1.00,0.88,1.00,0.96  Learning_rate:0.0001
Iter:4540  Loss:0.028  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0001
Iter:4560  Loss:0.038  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:4580  Loss:0.152  Accuracy:1.00,0.96,0.96,0.88  Learning_rate:0.0001
Iter:4600  Loss:0.013  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4620  Loss:0.039  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:4640  Loss:0.029  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:4660  Loss:0.040  Accuracy:1.00,1.00,0.96,0.96  Learning_rate:0.0001
Iter:4680  Loss:0.014  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4700  Loss:0.085  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0001
Iter:4720  Loss:0.016  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4740  Loss:0.020  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:4760  Loss:0.022  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4780  Loss:0.008  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4800  Loss:0.035  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0001
Iter:4820  Loss:0.011  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4840  Loss:0.069  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4860  Loss:0.006  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4880  Loss:0.079  Accuracy:1.00,1.00,0.88,1.00  Learning_rate:0.0001
Iter:4900  Loss:0.031  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4920  Loss:0.039  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4940  Loss:0.008  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:4960  Loss:0.020  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:4980  Loss:0.009  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5000  Loss:0.006  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5020  Loss:0.007  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5040  Loss:0.073  Accuracy:1.00,0.96,0.96,1.00  Learning_rate:0.0001
Iter:5060  Loss:0.001  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5080  Loss:0.019  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:5100  Loss:0.026  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5120  Loss:0.042  Accuracy:0.96,0.96,1.00,1.00  Learning_rate:0.0001
Iter:5140  Loss:0.035  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:5160  Loss:0.002  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5180  Loss:0.005  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5200  Loss:0.043  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:5220  Loss:0.063  Accuracy:1.00,1.00,0.92,1.00  Learning_rate:0.0001
Iter:5240  Loss:0.022  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5260  Loss:0.109  Accuracy:0.96,1.00,0.92,0.96  Learning_rate:0.0001
Iter:5280  Loss:0.019  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:5300  Loss:0.006  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5320  Loss:0.012  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5340  Loss:0.024  Accuracy:0.96,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5360  Loss:0.013  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5380  Loss:0.015  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5400  Loss:0.020  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5420  Loss:0.061  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5440  Loss:0.038  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5460  Loss:0.057  Accuracy:1.00,0.96,0.96,0.92  Learning_rate:0.0001
Iter:5480  Loss:0.005  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5500  Loss:0.041  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:5520  Loss:0.036  Accuracy:0.96,1.00,1.00,0.96  Learning_rate:0.0001
Iter:5540  Loss:0.022  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5560  Loss:0.003  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5580  Loss:0.004  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5600  Loss:0.015  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5620  Loss:0.021  Accuracy:1.00,0.96,1.00,1.00  Learning_rate:0.0001
Iter:5640  Loss:0.002  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5660  Loss:0.004  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5680  Loss:0.029  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:5700  Loss:0.022  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:5720  Loss:0.044  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5740  Loss:0.044  Accuracy:1.00,1.00,0.92,0.96  Learning_rate:0.0001
Iter:5760  Loss:0.013  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5780  Loss:0.044  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:5800  Loss:0.004  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5820  Loss:0.028  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5840  Loss:0.011  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5860  Loss:0.039  Accuracy:1.00,1.00,1.00,0.96  Learning_rate:0.0001
Iter:5880  Loss:0.005  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5900  Loss:0.061  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:5920  Loss:0.003  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5940  Loss:0.014  Accuracy:0.96,1.00,1.00,1.00  Learning_rate:0.0001
Iter:5960  Loss:0.051  Accuracy:1.00,1.00,0.92,1.00  Learning_rate:0.0001
Iter:5980  Loss:0.015  Accuracy:1.00,1.00,0.96,1.00  Learning_rate:0.0001
Iter:6000  Loss:0.010  Accuracy:1.00,1.00,1.00,1.00  Learning_rate:0.0000

"""