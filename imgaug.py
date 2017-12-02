# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:28:07 2017

@author: admin
"""

import sys
import skimage
import inspect
import numpy as np
from scipy import ndimage
from skimage import transform

MAX_IMG_VALUE = 65535 #图像数据的最大取值

ALL_AUG_METHODS = {}

def flipud(img):
    return np.flipud(img)
    
def fliplr(img):
    return np.fliplr(img)
    
def rotate90(img):
    ang = np.random.randint(0, 4)
    return np.rot90(img, ang)
    
def add(img, low=-100, high=100):
    assert img[..., -1].max() <= 1 #最后一通道是标签，不要动
    img = img.copy().astype(np.float32)
    for k in range(img.shape[-1]-1):
        img[..., k] += np.random.randint(low, high)
    img[..., :-1] = np.clip(img[...,:-1], 0, MAX_IMG_VALUE)
    return img

def mul(img, low=0.8, high=1.2):
    assert img[..., -1].max() <= 1
    img = img.copy().astype(np.float32)
    for k in range(img.shape[-1]-1):
        img[..., k] *= np.random.uniform(low, high)
    img[..., :-1] = np.clip(img[...,:-1], 0, MAX_IMG_VALUE)
    return img

def gaussian_noise(img, loc=0, scale=0.1*MAX_IMG_VALUE):
    assert img[..., -1].max() <= 1
    img = img.copy().astype(np.float32)
    noise = np.random.normal(loc, scale, img.shape[:-1]+(img.shape[-1]-1,))
    img[..., :-1] += noise
    img[..., :-1] = np.clip(img[..., :-1], 0, MAX_IMG_VALUE)
    return img
    
def gaussian_blur(img, sig_low=0.0, sig_high=3.0):
    img = img.copy().astype(np.float32)
    # 每个通道都使用同样的filter
    sig = np.random.uniform(sig_low, sig_high)
    for k in range(img.shape[-1]):
        img[..., k] = ndimage.gaussian_filter(img[..., k], sig)
    #确保最后一通道的标签属于0或1
    img[..., -1] = np.clip(np.round(img[..., -1]), 0, 1)
    img[:] = np.clip(img[:], 0, MAX_IMG_VALUE)
    return img
    
def contrast_normal(img, alpha_low=0.5, alpha_high=1.5):
    img = img.copy().astype(np.float32)
    for k in range(img.shape[-1]-1):
        alpha = np.random.uniform(alpha_low, alpha_high)
        half = MAX_IMG_VALUE//2
        img[..., k] = alpha*(img[..., k]-half) + half
    img[..., :-1] = np.clip(img[..., :-1], 0, MAX_IMG_VALUE)
    return img

def zoom(img, ratio=0.08):
    shift_x = int(round(img.shape[0]/2))
    shift_y = int(round(img.shape[1]/2))
    scale_x = scale_y = np.random.uniform(-ratio, ratio) + 1.0
    matrix_to_topleft = transform.SimilarityTransform(
        translation=[-shift_x, -shift_y])
    matrix_transforms = transform.AffineTransform(
        scale=(scale_x, scale_y))
    matrix_to_center = transform.SimilarityTransform(
        translation=[shift_x, shift_y])
    matrix = (matrix_to_topleft + matrix_transforms +
              matrix_to_center)
    matrix = matrix.inverse
    img = img.copy()
    img = skimage.img_as_float(img.astype(np.uint16))
    img = transform.warp(img, matrix, mode='constant', cval=0)
    img = skimage.img_as_uint(img)
    assert not np.logical_and(img[..., -1]>0, img[..., -1] < 1).any()
    return img.astype(np.float32)

def translate(img, ratio=0.03):
    # 只translate 2015年的
    trans_row_bound = int(img.shape[0] * ratio)
    trans_col_bound = int(img.shape[1] * ratio)
    translation = (
        np.random.randint(-trans_row_bound, trans_row_bound),
        np.random.randint(-trans_col_bound, trans_col_bound)
    )
    tf = transform.SimilarityTransform(translation=translation)
    img = img.copy()
    img = img.astype(np.uint16)
    img = skimage.img_as_float(img)
    t = transform.warp(img[..., :4], tf, mode='constant', cval=0)
    img[..., :4] = t
    return skimage.img_as_uint(img).astype(np.float32)
    
def _show_img(name, ori_img, new_img):
    ori_img = ori_img.astype(np.uint16)
    new_img = new_img.astype(np.uint16)
    plt.subplot(2,3,1)
    plt.suptitle(name, fontsize=16)
    plt.imshow(skimage.img_as_ubyte(ori_img[..., :3]))
    plt.title('Ori 2015')
    plt.subplot(2,3,2)
    plt.imshow(skimage.img_as_ubyte(ori_img[..., 4:7]))
    plt.title('Ori 2017')
    plt.subplot(2,3,3)
    plt.imshow(ori_img[..., -1])
    plt.title('Ori Label')
    
    plt.subplot(2,3,4)
    plt.imshow(skimage.img_as_ubyte(new_img[..., :3]))
    plt.title('New 2015')
    plt.subplot(2,3,5)
    plt.imshow(skimage.img_as_ubyte(new_img[..., 4:7]))
    plt.title('New 2017')
    plt.subplot(2,3,6)
    plt.imshow(new_img[..., -1])
    plt.title('New Label')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    b = np.load('../../input/mark/p1p2_e2e_rgbn_1103/26_2_2629#5472_p1.npy')
    img = b
    img = img.astype(np.float32)
    all_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    for name, f in all_functions:
        if name[0] != '_':
            t = f(img)
            _show_img(name, img, t)
else:
    all_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    assert len(ALL_AUG_METHODS) == 0
    for name, f in all_functions:
        if name[0] != '_':
            ALL_AUG_METHODS[name] = f