# -*- coding: utf-8 -*-
"""
对结果进行后处理，包括填补小面积空洞、移除小面积误报区。

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import utils
import numpy as np
import skimage.io
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_holes

def post_proc(img, output_path, run_name, min_size=2000, area=50):
    t0 = time.time()
    print('Remove small holes: %d.'%min_size)
    ind = remove_small_holes(label(img), min_size=min_size, connectivity=img.ndim)
    
    print('Remove samll region: %d.'%area)
    img = ind.astype(np.uint8)
    lab_arr = label(img)
    lab_atr = regionprops(lab_arr)
    
    def fun(atr):
        if atr.area <= area:
            min_row, min_col, max_row, max_col = atr.bbox
            t = lab_arr[min_row:max_row, min_col:max_col]
            t[t==atr.label] = 0

    list(map(fun, lab_atr))
    ind = lab_arr > 0
    view = np.zeros(img.shape, np.uint8)
    view[ind] = 255
    
    view_name = 'view-hole%d-area%d-%s.tif'%(min_size, area, run_name)
    sub = ind.astype(np.uint8)
    sub_name = 'submit-hole%d-area%d-%s.tiff'%(min_size, area, run_name)
    utils.save_data_as_zip('postproc-%s'%run_name, output_path, 
                     [sub_name, view_name],
                     [sub, view])
    print('Post process time cost: %0.2fmin.'%((time.time()-t0)/60))
    
if __name__ == '__main__':
    output_path = '../../output/'
    run_name = 'test'
    img = skimage.io.imread('../../output/test.tiff')
    post_proc(img, output_path, run_name)
