#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> on 2018/2/6 10:50


#https://www.zybuluo.com/hanbingtao/note/476663
#手写数字识别

#数据下载
# from tensorflow.examples.tutorials.mnist import input_data
# mnist=input_data.read_data_sets('/tmp/',one_hot=True)
#或者：download:http://yann.lecun.com/exdb/mnist/

#文件路径
import os

base_path='/Users/yexianyong/Downloads/machine_learning/mnist'
training_images_path=os.path.join(base_path,'train-images-idx3-ubyte')
training_labels_path=os.path.join(base_path,'train-labels-idx1-ubyte')
test_images_path=os.path.join(base_path,'t10k-images-idx3-ubyte')
test_labels_path=os.path.join(base_path,'t10k-labels-idx1-ubyte')

# 读取文件
# http://blog.csdn.net/simple_the_best/article/details/75267863
import struct
import numpy as np


def load_mnist(images_path, labels_path, kind='train'):
	"""Load MNIST data from path"""
	
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8)
	
	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
	
	return images, labels


images, labels = load_mnist(training_images_path, training_labels_path)


#构造训练相关数据
#给images添加bias：1

#inputs
inputs=np.zeros((images.shape[0],images.shape[1]+1))
inputs[:,0:1]=1#为数据添加bias 1
inputs[:,1:]=images[:,:]

#超参数确定
#Three layers
#input layer nodes:785个(已添加bias)
#hidden layer nodes:301个(已添加bias)
#output layer nodes:10个

#weights不能为0
w12=np.random.uniform(-0.1,0.1,(301,inputs.shape[1]))
w23=np.random.uniform(-0.1,0.1,(10,301))

#学习率
µ=0.0001


# 计算
# 激活函数sigmoid
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


def get_result(vec):
	max_value_index = 0
	max_value = 0
	for i in range(len(vec)):
		if vec[i] > max_value:
			max_value = vec[i]
			max_value_index = i
	return max_value_index


def softmax(vec):
	return vec / np.sum(vec)


for x, y in zip(inputs, labels):
	
	# 计算输出值
	z2 = w12.dot(x)
	a2 = sigmoid(z2)
	
	z3 = w23.dot(a2)
	a3 = softmax(z3)
	#     print('output:',a3)
	
	# 反向传播
	#######################
	label = np.zeros(10)
	label[y] = 1
	#     print('lable:',label)
	#     print('output:',get_result(a3),'label:',y)
	
	delta3 = a3 - label
	#     print('delta3:',delta3)
	
	# 更新w23
	w23 = w23 + µ * delta3.reshape(len(delta3), 1).dot(a2.reshape(1, len(a2)))
	
	# 计算a2节点误差delta2
	delta2 = a2 * (1 - a2) * w23.T.dot(delta3)
	
	# 更新w12
	#     print(x.shape,w12.shape,delta2.shape)
	w12 = w12 + µ * delta2.reshape(len(delta2), 1).dot(x.reshape(1, len(x)))
	
	#计算代价函数值：
	J=0.5*np.sum(np.array(list(map(lambda x,y:np.square(x-y),a3,label))))
	print(J)
	

# 测试
test_images, test_labels = load_mnist(test_images_path, test_labels_path)
# 数据预处理
# inputs
test_inputs = np.zeros((test_images.shape[0], test_images.shape[1] + 1))
test_inputs[:, 0:1] = 1  # 为数据添加bias 1
test_inputs[:, 1:] = test_images[:, :]

error_counts = 0
for x, y in zip(test_inputs, test_labels):
	
	# 计算输出值
	z2 = w12.dot(x)
	a2 = sigmoid(z2)
	
	z3 = w23.dot(a2)
	a3 = softmax(z3)
	
	if int(get_result(a3)) != int(y):
		error_counts += 1

print(error_counts / len(test_labels))