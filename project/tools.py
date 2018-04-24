#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by yetongxue<yeli.studio@qq.com> 
# 2018/4/2

import os

def makedir_logs(dir_name):
	"""
	传入文件名，以文件名创建文件夹
	"""
	path=os.path.join(os.getcwd(),'logs/{}'.format(dir_name))
	if not os.path.exists(path):
		os.makedirs(path)
		print('Created logs dir:{}'.format(dir_name))
	return path
	

def makedir(path):
	created=False
	if not os.path.exists(path):
		os.makedirs(path)
		created=True
		print('Created dir:{}'.format(path))
	return created,path


def delete_dir_file(path,name):
	"""
	删除文件夹需要输入全名，文件则部分即可
	"""
	for root,dirs,files in os.walk(path):
		for dir in dirs:
			if dir==name:
				os.rmdir(os.path.join(root,dir))
		for file in files:
			if file.find(name)!=-1:
				os.remove(os.path.join(root, file))