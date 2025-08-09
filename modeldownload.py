#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/3/19 10:44
@source from: 
"""
#Qwen-1_8B-Chat-Int8
#模型下载
print('debug')
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen-1_8B-Chat',cache_dir='/home/model/public/real_zhangguowen/models')