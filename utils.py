# -*- coding: utf-8 -*-
"""
    公用的工具函数
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import zipfile

import skimage.io
import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K

def read_data(path, using_cache=True):
    """
    从oss或本地读数据。
    
    Args:
      path: 路径
      using_cache: 当路径为oss路径时，using_cache指明是否使用当前目录已有的同名文件。
      
    Returns:
      返回读取的数据，目前支持读取.npy、.tiff、.tif、.jpg。
    """
    name = path
    if not is_py3():
        _, name = os.path.split(path)
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if not is_py3() and (not tf.gfile.Exists(name) or not using_cache):
        # 在PAI上运行时，需要将文件读取到当前目录，之后才能用其他包读取文件
        print('Copy %s to %s.'%(path, name))
        tf.gfile.Copy(path, name, overwrite=True)
    if ext == '.tif' or ext == '.tiff' or ext == '.jpg':
        return skimage.io.imread(name)
    elif ext == '.npy':
        return np.load(name)
    else:
        print("Unrecognized format '%s', return path '%s'." % (ext, name))
        return name

def save_data(direc, name, data, over_write=True):
    """
    将数据保存至本地或者oss
    
    Args:
      direc: 目录
      name: 文件名(包含后缀名)
      data: 待保存的数据
      over_write: 是否覆盖已存在的同名文件
      
    Returns:
      本地运行返回文件的路径，PAI上运行，返回文件名
    """
    is_oss = 'oss' == direc[:3]
    _, ext = os.path.splitext(name)
    ext = ext.lower()
    save_path = name if is_oss else os.path.join(direc, name)
    if ext in ['.tif', '.tiff', '.jpg', '.jpeg']:
        img = Image.fromarray(data)
        img.save(save_path)
    elif ext == '.npy':
        np.save(save_path, data)
    else:
        raise ValueError('Unsupported format: %s.'%ext)
    if is_oss:
        oss_path = os.path.join(direc, save_path)
        print('Copy %s to %s.'%(save_path, oss_path))
        tf.gfile.Copy(save_path, oss_path, overwrite=over_write)
    return save_path

def save_data_as_zip(zip_name, direc, names, datas):
    """
    将一组数据打包存成zip存至本地或oss
    
    Args:
      zip_name: zip文件名(不包含后缀名)
      direc: 保存目录
      names: 文件名列表(包含后缀名)
      datas: 对应的数据列表
      
    """
    new_names = []
    for n, d in zip(names, datas):
        t = save_data(direc, n, d)
        new_names.append(t)
    
    zip_path = '%s.zip'%zip_name
    if is_py3():
        zip_path = os.path.join(direc, zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in new_names:
            if is_py3():
                _, arc_name = os.path.split(p)
            else:
                arc_name = p
            zf.write(p, arc_name)#p中带有目录时，必须指定arc_name，不然生成的zip是空的
    # 删除临时文件
    for p in new_names:
        print('Remove %s.'%p)
        tf.gfile.Remove(p)

    if not is_py3():
        oss_path = os.path.join(direc, zip_path)
        print('Backup %s to %s.'%(zip_path, oss_path))
        tf.gfile.Copy(zip_path, oss_path, overwrite=True)

def copy_from_oss(oss_path_list, using_cache=True):
    """
    从oss批量拷贝文件到当前目录。
    
    Args:
      oss_path_list: 文件路径列表
      using_cache: 是否使用已存在的同名文件
      
    Returns:
      文件名列表
    """
    path_list = []
    for k in oss_path_list:
        _, name = os.path.split(k)
        path_list.append(name)
    for a, b in zip(oss_path_list, path_list):
        if not tf.gfile.Exists(a):
            print("'%s' not found. Skip! ", file=sys.stderr)
            continue
        print('Copy %s to %s.'%(a, b))
        tf.gfile.Copy(a, b, overwrite=not using_cache)
    return path_list
    
def copy_to_oss(oss_dir, path_list):
    """
    从本地批量拷贝文件至oss
    
    Args:
      oss_dir: oss目录
      path_list：文件名列表
    """
    for k in path_list:
        oss_path = os.path.join(oss_dir, k)
        print('Backup %s to %s.'%(k, oss_path))
        tf.gfile.Copy(k, oss_path, overwrite=True)

def make_model(name, input_shape):
    """
    根据名称创建网络
    
    Args:
      name: 网络名，目前只支持unet
      input_shape: 输入数据的shape，(width, height, depth)
      
    Returns:
      创建好的网络
    """
    if name == 'unet':
        if input_shape[0] == 128:
            from unet import get_unet_128
            model = get_unet_128(input_shape, 2)
        elif input_shape[0] == 256:
            from unet import get_unet_256
            model = get_unet_256(input_shape, 2)
        elif input_shape[0] == 512:
            from unet import get_unet_512
            model = get_unet_512(input_shape, 2)
        else:
            raise ValueError('Unsupported size %d in unet.'%input_shape[0])
    else:
        raise ValueError("Unrecognized model '%s'."%name)
    return model

def is_py3():
    return sys.version_info >= (3, 0)

class SampleCounter(object):
    """
    统计输入数据切分成训练样本后的数量。用于计算step_per_epoch和validation_steps。
    """
    def __init__(self, npy_list, target_size, batch_size=32, aug=1):
        """
        Args:
          npy_list: 输入数据路径列表
          target_size: 窗口大小
          batch_size: batch size
          aug: 应用数据增强后数据扩大倍数
        """
        self.data = {}
        self.batch_size = batch_size
        target_size = target_size[0]
        gap = target_size // 2
        for e in npy_list:
            x = np.load(e)
            r, c = x.shape[:2]
            if r < target_size or c < target_size:
                print('%s 太小.'%e, file=sys.stderr)
                self.data[e] = 0
            else:
                t = math.ceil((r-target_size)/gap)*math.ceil((c-target_size)/gap)
                self.data[e] = t*aug
    
    def count_sample(self, npy_list):
        """
        npy_list切分成训练样本后的数量，应是构造函数里npy_list的子集
        """
        cnt = 0
        for p in npy_list:
            if p not in self.data:
                print('%s not found!'%p, file=sys.stderr)
            else:
                cnt += self.data[p]
        return int(round(cnt/self.batch_size))

def official_score(sub_img, input_dir):
    """
    官方计算f1的方式。已用第一阶段官方的标准答案验证过。
    """
    truth = read_data(os.path.join(input_dir, 'answer_complete.tif'))
    pred = sub_img
    
    pos = truth == 1
    neg = truth == 2
    pred_pos = pred > 0
    
    TP=sum(sum(np.logical_and(pos,pred_pos)))
    TP_FP=sum(sum(np.logical_and(np.logical_or(pos,neg),pred_pos)))
    precision=float(TP)/float(TP_FP)
    
    TP_FN=sum(sum(pos))
    recall=TP/TP_FN
    f1=2*precision*recall/(precision+recall)
    print('Official F1: %0.3f.'%f1)

def f1_score(y_true, y_pred, smooth=1):
    """
    f1 score，用于训练过程中选择模型
    """
    y_true = y_true[:,:,:,-1]
    y_pred = y_pred[:,:,:,-1]
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    f1_score = (2*c1+smooth)/(c2+c3+smooth)
    return f1_score

def dice_coef(y_true, y_pred, smooth=1, weight=1):
    """
    加权后的dice coefficient
    """
    y_true = y_true[:,:,:,-1]
    y_pred = y_pred[:,:,:,-1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight*K.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    目标函数
    """
    return 1-dice_coef(y_true, y_pred)

