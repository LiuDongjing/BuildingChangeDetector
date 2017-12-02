# -*- coding: utf-8 -*-
"""
对原图像进行预处理：消除拼接痕迹，把图像的数据范围拉到0-65535
"""

import os
import glob
import time
import tqdm
import numpy as np
import skimage.io

def denoise_image(img, mask, smooth=False, radius=3):
    """
    针对图像的一个通道进行预处理操作。只对mask所指定的拼接区域操作，
    即均值拉到0.5，然后范围扩大到0-65535.
    
    Args:
      img: 原图像的一个通道
      mask: 一个拼接区域的mask
      smooth: 是否对拼接边界进行平滑处理
      radius: 平滑边界时邻域半径
      
    Returns:
      预处理后的该通道的图像
    """
    assert img.dtype == np.uint16
    t = img.astype(np.float32)
    maxv = np.percentile(t[mask], 99)
    minv = np.percentile(t[mask], 1)
    t[mask] = (t[mask]-minv)/(maxv-minv)
    meanv = t[mask].mean()
    t[mask] += (0.5-meanv)
    t[mask] *= 65535
    t[t<0] = 0
    t[t>65535] = 65535
    img = t.astype(np.uint16)
    if not smooth:
        return img
    bnd = np.hstack([mask[:, 1:], mask[:, -1].reshape((-1,1))])
    bnd = np.logical_xor(mask, bnd)
    ret = img.copy()
    for x in tqdm.tqdm(range(bnd.shape[0])):
        for y in range(bnd.shape[1]):
            if not bnd[x,y]:
                continue
            lx = max(0, x-radius)
            hx = min(bnd.shape[0]-1, x+radius)
            ly = max(0, y-radius)
            hy = min(bnd.shape[1]-1, y+radius)
            sm = img[lx:hx, ly:hy].mean()
            ret[x, y] = np.int(round(sm))
    return ret

def synthesis(path, mask_path, year, phase='p4'):
    """
    对原图进行预处理
    
    Args:
      path: 原图路径
      mask_path: mask图像路径
      year: '2015' 或 '2017'
      phase: 比赛阶段
    """
    masks = []
    mask_paths = glob.glob(os.path.join(mask_path, '*%s*'%year))
    for p in mask_paths:
        m = skimage.io.imread(p)
        masks.append(m[..., -1] > 0)
    print('%s: %d masks.'%(year, len(masks)))
    ori_img = skimage.io.imread(path)
    ori_img = ori_img[:,:,[2,1,0,-1]]
    d, n = os.path.split(path)
    syn_img = np.zeros(ori_img.shape, ori_img.dtype)
    for k in range(ori_img.shape[-1]):
        #针对每个通道进行预处理
        t = ori_img[..., k].copy()
        for z in range(len(masks)):
            print('Channel: %d, Mask: %s.'%(k, mask_paths[z]))
            t = denoise_image(t, masks[z], smooth=False)
        syn_img[:,:,k] = t
    skimage.io.imsave(os.path.join(d,
                        '%s%s-denoise-rgbn.tif'%(year, phase)), syn_img)
    skimage.io.imsave(os.path.join(d,
                        '%s%s-denoise-rgb.tif'%(year, phase)), syn_img[...,:-1])

if __name__ == '__main__':
    path15 = '../../input/origin/2015p3-raw.tif'
    path17 = '../../input/origin/2017p3-raw.tif'
    phase = 'test'
    mask_path = '../../input/%s-mask/'%phase
    t0 = time.time()
    synthesis(path15, mask_path, '2015', phase=phase)
    synthesis(path17, mask_path, '2017', phase=phase)
    print('Synthesis time cost %0.2fmin.'%((time.time()-t0)/60))

