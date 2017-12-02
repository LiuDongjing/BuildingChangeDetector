# -*- coding: utf-8 -*-
"""
    根据最佳权重生成预测结果。代码和end2end_predict.py基本一致
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import tensorflow as tf

import configs
import utils
import tqdm

def predict_x(batch_x, model):
    """
    预测一个batch的数据
    """
    batch_y = model.predict(batch_x)
    return batch_y

def make_prediction_img(x, target_size, batch_size, predict):
    """
    滑动窗口预测图像。
    
    每次取target_size大小的图像预测，但只取中间的1/4，这样预测可以避免产生接缝。
    """
    # target window是正方形，target_size是边长
    quarter_target_size = target_size // 4
    half_target_size = target_size // 2

    pad_width = (
        (quarter_target_size, target_size),
        (quarter_target_size, target_size),
        (0, 0))

    # 只在前两维pad
    pad_x = np.pad(x, pad_width, 'constant', constant_values=0)
    pad_y = np.zeros(
        (pad_x.shape[0], pad_x.shape[1], 2),
        dtype=np.float32)

    def update_prediction_center(one_batch):
        """根据预测结果更新原图中的一个小窗口，只取预测结果正中间的1/4的区域"""
        wins = []
        for row_begin, row_end, col_begin, col_end in one_batch:
            win = pad_x[row_begin:row_end, col_begin:col_end, :]
            win = np.expand_dims(win, 0)
            wins.append(win)
        x_window = np.concatenate(wins, 0)
        y_window = predict(x_window)#预测一个窗格
        for k in range(len(wins)):
            row_begin, row_end, col_begin, col_end = one_batch[k]
            pred = y_window[k, ...]
            y_window_center = pred[
                quarter_target_size:target_size - quarter_target_size,
                quarter_target_size:target_size - quarter_target_size,
                :] #只取预测结果中间区域

            pad_y[
               row_begin + quarter_target_size:row_end - quarter_target_size,
               col_begin + quarter_target_size:col_end - quarter_target_size,
               :] = y_window_center #更新也只更新一半，不会重复更新一个地方

    # 每次移动半个窗格
    batchs = []
    batch = []
    for row_begin in range(0, pad_x.shape[0], half_target_size):
        for col_begin in range(0, pad_x.shape[1], half_target_size):
            row_end = row_begin + target_size
            col_end = col_begin + target_size
            if row_end <= pad_x.shape[0] and col_end <= pad_x.shape[1]:
                batch.append((row_begin, row_end, col_begin, col_end))
                if len(batch) == batch_size:
                    batchs.append(batch)
                    batch = []
    if len(batch) > 0:
        batchs.append(batch)
        batch = []
    for bat in tqdm.tqdm(batchs, desc='Batch pred'):
        update_prediction_center(bat)
    y = pad_y[quarter_target_size:quarter_target_size+x.shape[0],
              quarter_target_size:quarter_target_size+x.shape[1],
              :]
    return y #原图像的预测结果

def predict_pair(model, path15, path17, options):
    qb15 = utils.read_data(path15, using_cache=False)
    qb15 = qb15.astype(np.float32)
    qb17 = utils.read_data(path17, using_cache=False)
    qb17 = qb17.astype(np.float32)
    x = np.concatenate([qb15, qb17], 2)
    x = x[:,:,options.use_chans]
    y_probs = make_prediction_img(
        x, options.target_size[0],options.batch_size,
        lambda xx: predict_x(xx, model))
    y_preds = np.argmax(y_probs, axis=2)#取概率最大的，0或者1
    return y_preds.astype(np.uint8)

def predict(model, options):
    print('Predicting...')
    t0 = time.time()
    y_preds = predict_pair(model,
        os.path.join(options.input_path, options.origin_15),
        os.path.join(options.input_path, options.origin_17),options)
    change = y_preds.astype(np.uint8)
    sub = change.copy()
    sub_name = 'submit-%s.tiff'%options.run_name
    change *= 255
    view_name = 'view-%s.tiff'%options.run_name
    utils.save_data_as_zip(options.run_name, options.output_path, 
                     [sub_name, view_name],
                     [sub, change])
    print('Prediction time cost: %0.2f(min).'%((time.time()-t0)/60))

if __name__ == '__main__':
    options = None
    if utils.is_py3():
        # Python3
        options = configs.get_config('local_config_end2end.json')
    else:
        # Python2
        options = configs.get_config(argparse.ArgumentParser())
    input_shape = (options.target_size[0], options.target_size[1],
                   len(options.use_chans))
    model = utils.make_model(options.model_name, input_shape)
    if utils.is_py3():
        p = os.path.join(options.output_path, options.weight_path)
        print("Load weights from '%s'."%p)
        model.load_weights(p)
    else:
        path = os.path.join(options.output_path, options.weight_path)
        if tf.gfile.Exists(path):
            print('Copy %s to %s.'%(path, options.weight_path))
            tf.gfile.Copy(path, options.weight_path, overwrite=True)
        else:
            print('There is not pretrained model: %s'%path)
        model.load_weights(options.weight_path)
    
    predict(model, options)