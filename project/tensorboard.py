#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/3/27

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os

LOG_DIR='./logs'

#清空logs下的文件
for root,dirs,files in os.walk(LOG_DIR,True):
	for f in files:
		os.remove(os.path.join(root,f))



#训练批次的次数
max_steps=1001

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#载入图片
embedding=tf.Variable(tf.stack(mnist.test.images),trainable=False,name='embedding')

#参数概要
def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)  # 平均值
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)  # 标准差
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)  # 直方图


# tensorboard-命名空间
with tf.name_scope('input'):
	# 设置变量
	x = tf.placeholder(tf.float32, [None, 784], name='x_input')
	y = tf.placeholder(tf.float32, [None, 10], name='y_input')


#显示图片
with tf.name_scope('input_reshape'):
	image_shaped_input=tf.reshape(x,[-1,28,28,1])#-1代表不确定的数字：多少张图片不确定
	tf.summary.image('input',image_shaped_input)



with tf.name_scope('layer'):
	with tf.name_scope('wight'):
		Weights = tf.Variable(tf.zeros([784, 10]), name='Weights')
		variable_summaries(Weights)
	with tf.name_scope('bias'):
		biases = tf.Variable(tf.zeros([10]), name='biases')
		variable_summaries(biases)

# 构造模型
with tf.name_scope('Wx_b'):
	prediction = tf.nn.softmax(tf.matmul(x, Weights) + biases)

# loss
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(prediction - y))
	tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 评估
with tf.name_scope('accuracy'):
	# correct_prediction是一个bool list
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
	with tf.name_scope('accuracy'):
		# 准确率
		"""
		tf.cast:转化类型 True->1.0;False->0.0
		tf.reduce_mean:求平均值，即准确率
		"""
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


#产生metadata文件
METADATA_PATH=os.path.join(LOG_DIR,'metadata.tsv')
if tf.gfile.Exists(METADATA_PATH):
	tf.gfile.Remove(METADATA_PATH)
with open(METADATA_PATH,'w') as f:
	labels=sess.run(tf.argmax(mnist.test.labels,1))
	for i in range(len(mnist.test.labels)):
		f.write(str(labels[i])+'\n')




# 合并所有summary
merged = tf.summary.merge_all()

# tenforboard-写入文件
file_writer = tf.summary.FileWriter('logs/', sess.graph)

#tensorboard可视化配置项
saver=tf.train.Saver()
config=projector.ProjectorConfig()
embed=config.embeddings.add()
embed.tensor_name=embedding.name
embed.metadata_path=METADATA_PATH
IMAGE_PATH=os.path.join(LOG_DIR,'mnist_sprite.png')
embed.sprite.image_path=IMAGE_PATH
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(file_writer,config)



"""
等程序运行完毕在终端执行一下命令启动tersorboard：
tensorboard --logdir=./logs/
"""

# 训练
for i in range(max_steps):
	
	run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata=tf.RunMetadata()
	
	batch_xs, batch_ys = mnist.train.next_batch(100)
	summary, _ = sess.run(
			[merged, train_step],
			feed_dict={x: batch_xs, y: batch_ys},
			options=run_options,
			run_metadata=run_metadata
	)
	
	# 训练的时候将summary加入到writer
	file_writer.add_summary(summary, i)
	
	if i % 10 == 0:
		res = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print(res)
		
		

#保存
SESS_PAHT=os.path.join(LOG_DIR,'model.ckpt')
saver.save(sess,SESS_PAHT,global_step=max_steps)
file_writer.close()
sess.close()
		

#执行完毕在终端执行：
"""
tensorboard --logdir=LOG_PATH
"""