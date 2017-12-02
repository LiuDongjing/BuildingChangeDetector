# -*- coding: utf-8 -*-
"""
生成数据集
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from generators import FileGenerator, Batch

class Datasets(FileGenerator):
    """
    用于得到train和validation的generator
    """
    def __init__(self, options):
        self.options = options
        if sys.version_info > (3, 0):
            super().__init__(options)
        else:
            super(Datasets, self).__init__(options)

    def label_to_one_hot_batch(self, label_batch):
        if label_batch.ndim == 4:
            label_batch = np.squeeze(label_batch, axis=3)

        nb_labels = 2
        shape = np.concatenate([label_batch.shape, [nb_labels]])
        one_hot_batch = np.zeros(shape)

        for label in range(nb_labels):
            one_hot_batch[..., label][label_batch == label] = 1.
        return one_hot_batch

    def get_file_size(self, file_ind):
        im = np.load(file_ind, mmap_mode='r')
        nb_rows, nb_cols = im.shape[0:2]
        return nb_rows, nb_cols

    def get_img(self, file_ind, window):
        im = np.load(file_ind, mmap_mode='r')
        ((row_begin, row_end), (col_begin, col_end)) = window
        img = im[row_begin:row_end, col_begin:col_end, :]
        return img
        
    def get_img_all(self, file_ind):
        img = np.load(file_ind, mmap_mode='r')
        return img

    def make_batch(self, img_batch):
        batch = Batch()
        batch.x = img_batch[..., self.options.use_chans]
        batch.y = self.label_to_one_hot_batch(
                img_batch[..., -1])
        return batch
