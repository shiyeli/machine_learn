#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/4/23


from captcha.image import ImageCaptcha  #生成验证码
import numpy as np
from PIL import Image #这个也可以生成验证码
import random,sys,string



#获取随机字符串
def get_random_string(length):
    return ''.join([random.choice(string.ascii_uppercase+string.ascii_lowercase+string.digits) for _ in range(length)])

#生成验证码
def gen_captcha_image():
    image=ImageCaptcha()
    captcha=image.generate(get_random_string(4))
    image.write()











