# -*- coding: utf-8 -*-
"""
    根据标注数据(.npy)生成训练和验证数据集，采用generator的形式生成数据。中间包含了
    数据增强以及打乱样本顺序的操作。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from imgaug import ALL_AUG_METHODS

class Batch(object):
    def __init__(self):
        self.x = None
        self.y = None

class FileGenerator(object):
    """
    生成样本的generator
    """
    def __init__(self, options):
        self.options = options
        self.train_file_inds = options.train_files
        self.validation_file_inds = options.validation_files
        self.test_file_inds = options.test_files

    def get_file_inds(self, split):
        if split == 'train':
            file_inds = self.train_file_inds
        elif split == 'validation':
            file_inds = self.validation_file_inds
        elif split == 'test':
            file_inds = self.test_file_inds
        else:
            raise ValueError('{} is not a valid split'.format(split))
        return file_inds

    def get_samples(self, gen, nb_samples):
        """
        组装一个batch的训练样本。
        
        Args:
          gen: 生成一张小图的generator
          nb_samples: batch size
          
        Returns:
          batch
        """
        samples = []
        # gen返回图像的一部分和图像id，图像一部分的shape为(row, col, channels+label)
        for i, sample in enumerate(gen):
            # batch维加在0位
            samples.append(np.expand_dims(sample, axis=0))
            if i+1 == nb_samples:
                break

        if len(samples) > 0:
            return np.concatenate(samples, axis=0)
        return None

    def split_image(self, row_num, col_num, target_size, gap):
        """
        将图像分割成大小为target_size的小图，返回每个小图的坐标。注意，这些坐标
        已被打乱。
        """
        if not gap:
            gap = target_size[0]//2
        rows = list(range(0, row_num-target_size[0], gap))
        if rows[-1]+target_size[0] < row_num:
            rows.append(row_num - target_size[0])
        cols = list(range(0, col_num-target_size[1], gap))
        if cols[-1]+target_size[1] < col_num:
            cols.append(col_num - target_size[1])
        np.random.shuffle(rows)
        np.random.shuffle(cols)
        return rows, cols

    def make_gap_img_generator(self, file_inds, target_size, gap=None):
        """
        生成一个小图的generator，小图的生成顺序是随机的，同一张大图里的小图不再相关。
        """
        if not gap:
            gap = target_size[0]//2
        img_index = []
        for ind in file_inds:
            img = self.get_img_all(ind)
            rows, cols = self.split_image(img.shape[0],
                                          img.shape[1], target_size, gap)
            for r in rows:
                for c in cols:
                    img_index.append((img, r, c))
        while True:
            np.random.shuffle(img_index)
            for img, r, c in img_index:
                t = img[r:r+target_size[0], c:c+target_size[1], :].copy()
                yield t

    def make_img_batch_generator(self, file_inds, target_size, batch_size):
        """
        生成一个batch样本的generator
        """
        gen = self.make_gap_img_generator(file_inds, target_size)
        while True:
            # 返回batch_size大小的数组，shape为(batch_size, row, col, channels+lables)
            batch = self.get_samples(gen, batch_size)
            if batch is None:
                raise StopIteration()
            yield batch

    def augment_img_batch(self, img_batch, augment_methods):
        """
        数据增强
        """
        imgs = []
        for sample_ind in range(img_batch.shape[0]):
            img = img_batch[sample_ind, ...]
            for m in augment_methods:
                if m not in ALL_AUG_METHODS:
                    print("'%s' is not a aug method."%m, file=sys.stderr)
                else:
                    if np.random.uniform() > 0.5:
                        img = ALL_AUG_METHODS[m](img)
            imgs.append(np.expand_dims(img, axis=0))
        img_batch = np.concatenate(imgs, axis=0)
        return img_batch

    def make_split_generator(self, split):
        """
        根据split生成对应的generator，split可以取'train'、'validation'和
        'test'
        """
        target_size = self.options.target_size
        batch_size = self.options.batch_size
        augment_methods = self.options.augment_methods
        
        def py2_map_generator(func, gen):
            """
            兼容python2的map函数
            """
            for x in gen:
                yield func(x)

        file_inds = self.get_file_inds(split)
        img_batch_gen = self.make_img_batch_generator(
            file_inds, target_size, batch_size)

        def transform_img_batch(x):
            img_batch = x.astype(np.float32)

            if augment_methods:
                img_batch = self.augment_img_batch(img_batch, augment_methods)

            batch = self.make_batch(img_batch)
            return batch.x, batch.y

        if sys.version_info > (3, 0):
            split_gen = map(transform_img_batch, img_batch_gen)
        else:
            split_gen = py2_map_generator(transform_img_batch, img_batch_gen)
        return split_gen