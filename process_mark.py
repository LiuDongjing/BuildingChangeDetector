# -*- coding: utf-8 -*-
"""
    根据手工标注的标签生成npy文件，每个npy文件保存了一个二维数组(width, height, 8+1)，
    前8个通道是图像数据，最后一个通道是标签
"""
import os
import sys
import glob
import json

import tqdm
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

def find_bnd(img):
    """
    从标签图像中找到标注的区域(矩形)，标签图像是与卫星图大小一致的RGBA图像，未标注的
    区域为黑色，标注的区域如果是房子变化则为红色，否则为透明(四通道数值均为0)。
    """
    r_s = -1
    r_e = -1
    c_s = -1
    c_e = -1
    fg = False
    r_m = -1
    c_m = -1
    for i in range(0, img.shape[0], 256):
        if fg:
            break
        for j in range(0, img.shape[1], 256):
            if img[i, j, 0] > 0 or img[i,j,3] == 0:
                r_m = i
                c_m = j
                fg = True
                break
    if r_m < 0 or c_m < 0:
        return r_s, r_e, c_s, c_e
    r_s = r_m
    c_s = c_m
    while img[r_s, c_m, 0] > 0 or img[r_s, c_m, 3] == 0:
        r_s -= 1
    r_s += 1
    while img[r_m, c_s, 0] > 0 or img[r_m, c_s, 3] == 0:
        c_s -= 1
    c_s += 1
    r_e = r_m+1
    c_e = c_m+1
    while img[r_e, c_m, 0] > 0 or img[r_e, c_m, 3] == 0:
        r_e += 1
    while img[r_m, c_e, 0] > 0 or img[r_m, c_e, 3] == 0:
        c_e += 1
    return r_s, r_e, c_s, c_e

def prepare_end2end_data(path15, path17, input_dir, output_dir, base=0, suffix='p2'):
    """
    将标注的标签图像转换成npy文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    im15 = skimage.io.imread(path15).astype(np.float32)
    im17 = skimage.io.imread(path17).astype(np.float32)
    masks = glob.glob(os.path.join(input_dir, '*.tif'))
    data_dir = output_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for mp in tqdm.tqdm(masks):
        msk = skimage.io.imread(mp)
        r_s, r_e, c_s, c_e = find_bnd(msk)
        if r_s < 0:
            print('%s无效!'%mp, file=sys.stderr)
            continue
        d15 = im15[r_s:r_e, c_s:c_e, :]
        d17 = im17[r_s:r_e, c_s:c_e, :]
        m = msk[r_s:r_e, c_s:c_e, 0]
        lab = m > 0
        lab = lab.astype(d15.dtype)
        lab = np.expand_dims(lab, 2)
        d = np.concatenate([d15, d17, lab], 2)
        vp = d.shape[0]//2
        hp = d.shape[1]//2
        lst = [d[:vp, :hp, :], d[vp:, :hp, :], d[:vp, hp:, :], d[vp:, hp:, :]]
        cord = [(r_s, c_s), (r_s+vp, c_s), (r_s, c_s+hp), (r_s+vp, c_s+hp)]
        _, n = os.path.split(mp)
        n, _ = os.path.splitext(n)
        for k in range(len(lst)):
            r, c = cord[k]
            dp = os.path.join(data_dir, '%d_%d_%d#%d_%s.npy'%(base, k, r, c, suffix))
            np.save(dp, lst[k])
        base += 1

def end2end_split(data_dir, splits=None):
    """
    将npy文件划分成训练集、验证集和测试集
    """
    if splits is None:
        splits = [0.7, 0.9]
    assert splits[0] < splits[1]
    dat_all = glob.glob(os.path.join(data_dir, '*.npy'))
    np.random.shuffle(dat_all)
    for k in range(len(dat_all)):
        _, dat_all[k] = os.path.split(dat_all[k])
    mp = {}
    sp1 = int(len(dat_all)*splits[0])
    sp2 = int(len(dat_all)*splits[1])
    mp['train'] = dat_all[:sp1]
    mp['validation'] = dat_all[sp1:sp2]
    mp['test'] = dat_all[sp2:]
    if data_dir[-1] == '/' or data_dir[-1] == '\\':
        _, dir_name = os.path.split(data_dir[:-1])
    else:
        _, dir_name = os.path.split(data_dir)
    with open(os.path.join(data_dir, '%s_train_val_test.json'%dir_name), 'w') as file:
        file.write(json.dumps(mp))

def end2end_data_view(input_dir, part='validation'):
    """
    查看训练集、验证集或测试集里的图像
    """
    mp = glob.glob(os.path.join(input_dir, '*.json'))[0]
    mp = json.load(open(mp))
    paths = mp[part]
    for p in paths:
        t = os.path.join(input_dir, p)
        x = np.load(t)
        im15 = skimage.img_as_ubyte(x[:,:,:3].astype(np.uint16))
        im17 = skimage.img_as_ubyte(x[:,:,4:7].astype(np.uint16))
        msk = x[:,:,-1].astype(np.uint8)
        msk *= 90
        msk += 255-90
        msk = np.expand_dims(msk, 2)
        im15 = np.concatenate([im15, msk],2)
        im17 = np.concatenate([im17, msk],2)
        plt.subplot(1,2,1)
        plt.imshow(im15)
        plt.title('2015')
        plt.suptitle(p)
        plt.subplot(1,2,2)
        plt.imshow(im17)
        plt.title('2017')
        plt.show()

if __name__ == '__main__':
    prepare_end2end_data(
        '../../input/origin/2015p2-denoise-rgbn.tif',
        '../../input/origin/2017p2-denoise-rgbn.tif',        
        '../../input/mark/p2_end2end_1102/', 
        '../../input/mark/p2_test/', base=0, suffix='p2')
    end2end_split('../../input/mark/p2_test/')
    end2end_data_view('../../input/mark/p2_test/', part='validation')