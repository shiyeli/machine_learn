#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/4/2

import os

def makedir_logs(dir_name):
	"""
	传入文件名，以文件名创建文件夹
	:return:
	:rtype:
	"""
	path=os.path.join(os.getcwd(),'logs/{}'.format(dir_name))
	if not os.path.exists(path):
		os.makedirs(path)
		print('Created logs dir:{}'.format(dir_name))
	return path
	




