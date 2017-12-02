# -*- coding: utf-8 -*-
"""
    训练网络。
    
    在本地运行，当前目录应有local_config_end2end.json配置文件；在PAI上运行，应制定
    配置文件，格式参考unet-end2end-rgbn.txt。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time
import json
import argparse

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import (Callback, ModelCheckpoint, CSVLogger,
                             LearningRateScheduler)

import utils
import configs
from datasets import Datasets
from end2end_predict import test

def get_initial_epoch(log_path):
    """
        从log文件中获取最近训练结束时的epoch，重新运行代码后会接着这个epoch继续训练。
    """
    initial_epoch = 0
    if tf.gfile.Exists(log_path):
        with open(log_path) as log_file:
            line_ind = -1
            for _, line in enumerate(log_file):
                line = line.strip()
                if line == '':
                    continue
                t = re.split(r',', line)[0]
                if not t.isdigit():
                    continue
                line_ind = 1+int(t)
            initial_epoch = line_ind
    return initial_epoch

class ModelBackup(Callback):
    """
        保存训练好的网络权重。
        
        PAI上运行代码时才会使用该功能。每个epoch结束后，都会将得到的最佳网络权重保存到
        自己的oss目录里。
    """

    def __init__(self, output_path, model_names):
        """
        Args:
          output_path: 保存网络权重的oss目录
          model_namse: 权重文件的名称
        """
        self.output_path = output_path
        self.model_names = model_names
        if sys.version_info > (3, 0):
            super().__init__()
        else:
            super(ModelBackup, self).__init__()

    def on_epoch_end(self, batch, logs=None):
        for e in self.model_names:
            if tf.gfile.Exists(e):
                path = os.path.join(self.output_path, e)
                print('Backup %s to %s.'%(e, path))
                tf.gfile.Copy(e, path, overwrite=True)
            else:
                print('%s not found!'%e, file=sys.stderr)
            
class TrainModel():
    """
        训练模型。
    """
    def __init__(self, options):
        """
        Args:
          options: configs.Config类的实例
        """
        self.options = options
        
        input_shape = (options.target_size[0], options.target_size[1],
                       len(options.use_chans))
        print('Making %s...'%options.model_name)
        if utils.is_py3(): #本地运行
            best_model = os.path.join(options.output_path, options.weight_path)
            self.model = utils.make_model(self.options.model_name, input_shape)
            if tf.gfile.Exists(best_model):
                print('Load weights from %s.'%best_model)
                self.model.load_weights(best_model)
        else: #PAI上运行
            path = os.path.join(options.output_path, options.weight_path)
            self.model = utils.make_model(self.options.model_name, input_shape)
            if tf.gfile.Exists(path):
                utils.copy_from_oss([path], using_cache=False)
                best_model = options.weight_path
                print('Load weights from %s.'%best_model)
                self.model.load_weights(best_model)
        
        self.log_path = '%s-log.txt'%options.run_name
        if utils.is_py3():
            self.log_path = os.path.join(options.output_path, self.log_path)
        else:
            utils.copy_from_oss(
                [os.path.join(options.output_path, self.log_path)],
                 using_cache=False)
        if self.options.init_epoch < 0:
            self.options.init_epoch = get_initial_epoch(self.log_path)

        self.metrics = [utils.f1_score]
        self.loss_function = utils.dice_coef_loss

    def make_callbacks(self):
        tmp_path = self.options.output_path
        if not utils.is_py3():
            tmp_path = ''
        best_model_checkpoint = ModelCheckpoint(monitor='val_f1_score', mode='max',
            filepath=os.path.join(tmp_path, options.weight_path), save_best_only=True,
            save_weights_only=True)
        logger = CSVLogger(self.log_path, append=True)
        callbacks = [best_model_checkpoint, logger]
        if not utils.is_py3():
            callbacks.append(ModelBackup(self.options.output_path,
                            [self.options.weight_path,
                             self.log_path]))

        if self.options.lr_epoch_decay: # 每个epoch都会降低lr
            def get_lr(epoch):
                w = epoch // 10
                lr = self.options.init_lr / (options.lr_epoch_decay ** w)
                if lr < 1e-10:
                    lr = 1e-10
                return lr
            callback = LearningRateScheduler(get_lr)
            callbacks.append(callback)

        return callbacks

    def train_model(self):
        t0 = time.time()
        print(self.model.summary())
        dataset = Datasets(self.options)
        train_gen = dataset.make_split_generator('train')
        validation_gen = dataset.make_split_generator('validation')
        optimizer = Adam(lr=self.options.init_lr, decay=0)
        self.model.compile(
            optimizer, loss=self.loss_function, metrics=self.metrics)
            
        callbacks = self.make_callbacks()
        self.model.fit_generator(
            train_gen,
            initial_epoch=self.options.init_epoch,
            steps_per_epoch=self.options.steps_per_epoch,
            epochs=self.options.epochs,
            validation_data=validation_gen,
            validation_steps=self.options.validation_steps,
            callbacks=callbacks)
            
        print('Training time cost: %0.2f(min).'%((time.time()-t0)/60))

if __name__ == '__main__':
    options = None
    t0 = time.time()
    train_files = []
    val_files = []
    test_files = []
    if utils.is_py3():
        # Python3
        options = configs.get_config('local_config_end2end.json')
        train_val = json.load(open(options.train_val_test_config))
        for e in train_val['train']:
            train_files.append(os.path.join(options.input_path, 
                                            options.data_path, e))
        for e in train_val['validation']:
            val_files.append(os.path.join(options.input_path, 
                                          options.data_path, e))
        for e in train_val['test']:
            test_files.append(os.path.join(options.input_path, 
                                          options.data_path, e))
    else:
        # Python2
        options = configs.get_config(argparse.ArgumentParser())
        train_val = json.load(open(options.train_val_test_config))
        
        # 将data_path里所有的npy文件拷贝到当前目录
        data = tf.gfile.Glob(os.path.join(
                              options.input_path, options.data_path, '*.npy'))
        utils.copy_from_oss(data, using_cache=False)
        
        for e in train_val['train']:
            train_files.append(e)
        for e in train_val['validation']:
            val_files.append(e)
        for e in train_val['test']:
            test_files.append(e)

    counter = utils.SampleCounter(train_files+val_files,
                                  options.target_size, options.batch_size,
                                  1)
    options.train_files = train_files
    options.validation_files = val_files
    options.test_files = test_files

    options.steps_per_epoch = counter.count_sample(options.train_files)
    options.validation_steps = counter.count_sample(options.validation_files)
    options.show()
    
    model = TrainModel(options=options)
    model.train_model()

    test(model.model, options)
    print('Total time cost: %0.2f(min).'%((time.time()-t0)/60))