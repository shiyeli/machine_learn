#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/7/20

import model
import tensorflow as tf
from tensorflow.contrib import slim
from PIL import Image
import matplotlib.pyplot as plt
import tools
import vgg
import os

TRAIN_DATA_DIR = '/Users/yexianyong/python/deep_learn/tensorflow_1.4.0/flower_photos/roses'

STYLE_IMAGE_PATH = 'images/star.jpg'
TEST_IMAGE_PATH = 'images/sleep.jpg'
VGG_MODEL_PATH = 'tmp/vgg_16.ckpt'
MODEL_SAVE = 'tmp/model/model.ckpt'


class Train(object):
    def __init__(self, sess):
        self.sess = sess
        self.batch_size = 4
        self.img_size = 256
        self.img_dim = 3
        self.gamma = 0.7
        self.lamda = 0.001
        self.load_model = False
        self.max_step = 20
        self.style_w = 10
        self.learn_rate_base = 0.0005
        self.learn_rate_decay = 0.9

    def build_model(self):
        train_imgs = tools.load_train_img(TRAIN_DATA_DIR, self.batch_size, self.img_size)
        style_imgs = tools.load_style_img(STYLE_IMAGE_PATH)

        with slim.arg_scope(model.arg_scope()):
            gen_img, variables = model.inference(train_imgs, reuse=False, name='transform')

            with slim.arg_scope(vgg.vgg_arg_scope()):
                gen_img_processed = [tf.image.per_image_standardization(image) for image in
                                     tf.unstack(gen_img, axis=0, num=self.batch_size)]

                f1, f2, f3, f4, exclude = vgg.vgg_16(tf.concat([gen_img_processed, train_imgs, style_imgs], axis=0))

                gen_f, img_f, _ = tf.split(f4, 3, 0)
                content_loss = tf.nn.l2_loss(gen_f - img_f) / tf.to_float(tf.size(gen_f))

                style_loss = model.styleloss(f1, f2, f3, f4)

                vgg_model_path = VGG_MODEL_PATH
                vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
                init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
                init_fn(self.sess)
                print("vgg's weights load done")

            self.gen_img = gen_img
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.content_loss = content_loss
            self.style_loss = style_loss * self.style_w
            self.loss = self.content_loss + self.style_loss
            self.learn_rate = tf.train.exponential_decay(self.learn_rate_base, self.global_step, 1,
                                                         self.learn_rate_decay, staircase=True)
            self.opt = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, global_step=self.global_step,
                                                                        var_list=variables)

        all_var = tf.global_variables()
        init_var = [v for v in all_var if 'vgg_16' not in v.name]
        init = tf.variables_initializer(var_list=init_var)
        self.sess.run(init)
        self.save = tf.train.Saver(var_list=variables)

    def train(self):
        print('start to training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                _, loss, step, cl, sl, learn_rate = self.sess.run(
                    [self.opt, self.loss, self.global_step, self.content_loss, self.style_loss, self.learn_rate])
                print('[{}/{}],loss:{}, content:{},style:{},learn_rate:{}'.format(self.max_step, step, loss, cl, sl,
                                                                                  learn_rate))

                if step % 10 == 0:
                    gen_img = self.sess.run(self.gen_img)
                    if not os.path.exists('tmp/gen_img'):
                        os.mkdir('tmp/gen_img')
                    tools.save_images(gen_img, 'tmp/gen_img/{0}.jpg'.format(step))

                if step % 10 == 0:
                    if not os.path.exists('tmp/model'):
                        os.mkdir('tmp/model')
                    self.save.save(self.sess, MODEL_SAVE, global_step=step)
                if step >= self.max_step:
                    break

        except tf.errors.OutOfRangeError:
            self.save.save(self.sess, os.path.join(os.getcwd(), 'tmp/model/finally_model.ckpt'))
        finally:
            coord.request_stop()
        coord.join(threads)

    def test(self):
        print ('test model')
        test_img = tools.load_test_img(TEST_IMAGE_PATH)
        test_img = self.sess.run(test_img)
        with slim.arg_scope(model.arg_scope()):
            gen_img, _ = model.inference(test_img, reuse=False, name='transform')

            vars = slim.get_variables_to_restore(include=['transform'])
            init_fn = slim.assign_from_checkpoint_fn('tmp/model/model.ckpt-50', vars)
            init_fn(self.sess)

            gen_img = self.sess.run(gen_img)
            tools.save_images(gen_img, 'images/test.jpg')


if __name__ == '__main__':
    with tf.Session() as sess:
        train_model = Train(sess)
        # train_model.build_model()
        # train_model.train()

        train_model.test()
