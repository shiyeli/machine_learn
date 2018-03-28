#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/3/27

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOG_DIR=os.path.join(os.getcwd(),'logs')


NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
#可视化图片数量
TO_EMBED_COUNT=3000

path_for_mnist_sprites =  os.path.join(LOG_DIR,'mnistdigits.png')
path_for_mnist_metadata =  os.path.join(LOG_DIR,'metadata.tsv')


#清空logs下的文件
for root,dirs,files in os.walk(LOG_DIR,True):
	for f in files:
		os.remove(os.path.join(root,f))



#训练批次的次数
max_steps=10000

#学习率
learn_rate=5e-1



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist_not_one_hot = input_data.read_data_sets("MNIST_data/", one_hot=False)


batch_xs,batch_ys=mnist_not_one_hot.train.next_batch(TO_EMBED_COUNT)

#载入图片的变量
embedding_var=tf.Variable(batch_xs,name=NAME_TO_VISUALISE_VARIABLE)

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
		Weights = tf.Variable(tf.truncated_normal([784, 10],stddev=0.1), name='Weights')
		variable_summaries(Weights)
	with tf.name_scope('bias'):
		biases = tf.Variable(tf.zeros([10])+0.1, name='biases')
		variable_summaries(biases)

# 构造模型
with tf.name_scope('Wx_b'):
	prediction = tf.nn.softmax(tf.matmul(x, Weights) + biases)

# loss
with tf.name_scope('loss'):
	# loss = tf.reduce_mean(tf.square(prediction - y))
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
	tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

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



# 合并所有summary
merged = tf.summary.merge_all()

# tenforboard-写入文件
file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

#建立 embedding projector
config=projector.ProjectorConfig()
embedding=config.embeddings.add()
embedding.tensor_name=embedding_var.name

#指定metadata位置
embedding.metadata_path=path_for_mnist_metadata
#指定sprite位置
embedding.sprite.image_path=path_for_mnist_sprites
embedding.sprite.single_image_dim.extend([28,28])
# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer=file_writer,config=config)

#保存：Tensorboard 会从保存的图形中加载保存的变量，所以初始化 session 和变量，并将其保存在 logdir 中
saver=tf.train.Saver()
saver.save(sess,os.path.join(LOG_DIR,'model.ckpt'),1)



#定义 helper functions
"""
**create_sprite_image:** 将 sprits 整齐地对齐在方形画布上
**vector_to_matrix_mnist:** 将 MNIST 的 vector 数据形式转化为 images
**invert_grayscale: **将黑背景变为白背景
"""
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))


    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


#保存 sprite image:将 vector 转换为 images，反转灰度，并创建并保存 sprite image

to_visualise = batch_xs
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')


# 保存 metadata
"""
将数据写入 metadata，因为如果想在可视化时看到不同数字用不同颜色表示，需要知道每个 image 的标签，
在这个 metadata 文件中有这样两列：”Index” , “Label”
"""

with open(path_for_mnist_metadata,'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(batch_ys):
        f.write("%d\t%d\n" % (index,label))





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
	
	if i % 1000 == 0:
		train_accuracy=sess.run(accuracy,feed_dict={x:mnist.train.images,y: mnist.train.labels})
		test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print('Test data accuracy:%s, train data accuracy:%s ' % (test_accuracy,train_accuracy))
		
		

#保存
SESS_PAHT=os.path.join(LOG_DIR,'model.ckpt')
saver.save(sess,SESS_PAHT,global_step=max_steps)
file_writer.close()
sess.close()
		

#执行完毕在终端执行：
"""
tensorboard --logdir=LOG_PATH
"""