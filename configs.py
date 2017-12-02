# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:26:30 2017

@author: admin

所有有关训练网络的参数。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import json
import tensorflow as tf

def get_config(parser_or_path):
    """
    获取一个Config对象。
       
    Args:
      parser_or_path: argparse.ArgumentParser对象或者json文件的路径；当在PAI上训练
        网络时，通过ArgumentParser接收命令行参数来来构建Config对象，当在本地训练网络
        时，通过json配置文件构建Config对象。
        
    Returns:
      Config对象
    """
    if isinstance(parser_or_path, str):
        return Config(parser_or_path)
    parser = parser_or_path

    # 参数的详细解释参考Config的定义
    # 必需参数
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--origin_15', type=str)
    parser.add_argument('--origin_17', type=str)
    parser.add_argument('--train_val_test_config', type=str)
    
    # 可选参数
    parser.add_argument('--model_name', type=str, default='unet')
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--target_size', type=int, default=128)
    parser.add_argument('--lr_epoch_decay', type=int, default=10)
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--init_epoch', type=int, default=-1)
    parser.add_argument('--augmentation', type=str, default='')
    parser.add_argument('--use_chans', type=int, default=255)
    
    args = parser.parse_known_args()
    return Config(vars(args[0]))

class Config(object):
    """
    配置信息
    """
    def __init__(self, path_or_dict):
        dat = path_or_dict
        if isinstance(dat, str):
            dat = json.loads(tf.gfile.GFile(dat).read())
            
    # 必需参数
        if 'input_path' not in dat:
            raise ValueError("'input_path' not in config file.")
        # 所有输入数据的根目录
        self.input_path = dat['input_path']
        
        if 'output_path' not in dat:
            raise ValueError("'output_path' not in config file.")
        # 输出数据的根目录
        self.output_path = dat['output_path']
        
        if 'train_val_test_config' not in dat:
            raise ValueError("'train_val_test_config' not in config file.")
        # 划分train、validation和test数据集的json配置文件名，该文件在当前目录
        self.train_val_test_config = dat['train_val_test_config']

        if 'origin_15' not in dat:
            raise ValueError("'origin_15 not in config file.'")
        # 2015年卫星图像相对于input_path的路径
        self.origin_15 = dat['origin_15']
        
        if 'origin_17' not in dat:
            raise ValueError("'origin_17 not in config file.'")
        # 2017年卫星图像相对于input_path的路径
        self.origin_17 = dat['origin_17']
        
        if 'data_path' not in dat:
            raise ValueError("'data_path not in config file.'")
        # 训练数据相对于input_path的路径
        self.data_path = dat['data_path']
        
    # 可选参数
        # 模型名称，根据名称构建对应的模型
        self.model_name = 'unet'
        if 'model_name' in dat:
            self.model_name = dat['model_name']
        
        self.init_lr = 1e-4
        if 'init_lr' in dat:
            self.init_lr = dat['init_lr']
        self.epochs = 20
        if 'epochs' in dat:
            self.epochs = dat['epochs']
        
        self.init_epoch = -1
        if 'init_epoch' in dat:
            self.init_epoch = dat['init_epoch']
            
        self.batch_size = 16
        if 'batch_size' in dat:
            self.batch_size = dat['batch_size']
        
        # 一次训练的名称，用于区分不同训练得到的网络权重
        self.run_name = 'unknown'
        if 'run_name' in dat:
            self.run_name = dat['run_name']
        self.run_name = '%s-%s'%(self.run_name, self.model_name)
        
        self.lr_epoch_decay = 10
        if 'lr_epoch_decay' in dat:
            self.lr_epoch_decay = dat['lr_epoch_decay']
            
        self.target_size = (128, 128)
        if 'target_size' in dat:
            self.target_size = (dat['target_size'], dat['target_size'])

        # 最佳网络权重保存路径，相对于output_path。
        if 'weight_path' not in dat or dat['weight_path'] is '':
            self.weight_path = '%s-best.h5'%self.run_name
        else:
            self.weight_path = dat['weight_path']
        
        # 数据增强的方法，不同的方法用'-'隔开，具体的有哪些方法参考imgaug.py
        self.augment_methods = None
        if 'augmentation' in dat and dat['augmentation'] is not '':
            self.augment_methods = re.split(r'-', dat['augmentation'])
        
        # 训练网络时使用输入图像的哪些channel，给的是一个整数，当使用某个channel时，
        # 对应的二进制位为1
        self.use_chans = 255
        if 'use_chans' in dat:
            self.use_chans = dat['use_chans']
        t = []
        x = [1,2,4,8,16,32,64,128,256,512,1024]
        for k in range(len(x)):
            if self.use_chans & x[k]:
                t.append(k)
        self.use_chans = t
        
    # 运行时参数
        
        self.steps_per_epoch = 0
        self.validation_steps = 0
        self.train_files = []
        self.validation_files = []
        self.test_files = []

    def show(self):
        print(10*'*', 'Configures', 10*'*')
        print('model name: ', self.model_name)
        print('epochs: ', self.epochs)
        print('batch size: ', self.batch_size)
        print('target size: ', self.target_size)
        print('steps per epoch: ', self.steps_per_epoch)
        print('validation steps: ', self.validation_steps)
        print('init lr: ', self.init_lr)
        print('use chans: ', self.use_chans)
        print('run name: ', self.run_name)
        print('init epoch: ', self.init_epoch)
        print('aug methods: ', self.augment_methods)
        print('weight path: ', self.weight_path)
        print(30*'*')