import SimpleITK as sitk
from subprocess import Popen
import subprocess
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import functools
from functools import partial
import sys
import os.path
from os import path as pathOs
import numpy as np
import tempfile
import shutil
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
import fileinput
import re
import subprocess
from toolz.itertoolz import groupby
import seaborn as sns
import einops
import matplotlib.pyplot as plt
import itertools
import SimpleITK as sitk
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import multiprocessing as mp
import functools
from functools import partial
import math
import time
import itertools
from pydicom.fileset import FileSet
from os import path as pathOs
from pathlib import Path
import toolz
from toolz.curried import pipe, map, filter, get
from toolz import curry
from os.path import basename, dirname, exists, isdir, join, split
import nnunetv2
import tempfile
import shutil
import re
from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import os
from subprocess import Popen
import subprocess
from skimage.segmentation import mark_boundaries
from scipy import ndimage
import torch
import xformers.components.attention.attention_patterns as AP
from xformers.components.attention.core import scaled_dot_product_attention
from xformers.components.attention._sputnik_sparse import SparseCS
import xformers.components.attention.attention_patterns as AP

from xformers.utils import (
    generate_matching_config,
    get_registry_decorator,
    import_all_modules,
)

from pathlib import Path
from typing import Any, Callable, Dict, Set, Union

from xformers.utils import (
    generate_matching_config,
    get_registry_decorator,
    import_all_modules,
)
from xformers.components.feedforward.base import Feedforward, FeedforwardConfig  # noqa
import h5py
from xformers.components.attention.core import _matmul_with_mask
from xformers.components.attention.core import _broadcast_batch
import torch 
import torch.nn as nn
import torch

h5_path="/workspaces/konwersjaJsonData/sparse_dat/sparse_masks.hdf5"

def _generate_nd_grid(*sizes):
    coords = [torch.arange(s) for s in sizes]
    return torch.meshgrid(*coords)
def get_init_last(c0,cc):
    diff=cc-c0
    beg=diff//2
    end=diff-beg
    return beg,end

def swin_attention_pattern_3D(H, W,D, window_size, shift_size=0, mask_other_shape=(0,0)):
    H0=H
    W0=W
    D0=D
    H= int(math.ceil(H/window_size))*window_size
    W= int(math.ceil(W/window_size))*window_size
    D= int(math.ceil(D/window_size))*window_size
    assert H % window_size == 0
    assert W % window_size == 0
    assert D % window_size == 0
    assert 0 <= shift_size < window_size, "shift_size must in 0-window_size"

    # input grid
    i, j ,k = _generate_nd_grid(H, W,D)
    i, j ,k = i + 0.5, j + 0.5, k + 0.5

    # anchors grid
    # if shift is present, add extra element to the grid
    # to account for the uneven partitioning
    extra = int(shift_size % window_size != 0)
    grid_h = H // window_size + extra
    grid_w = W // window_size + extra
    grid_d = D // window_size + extra

    ii, jj ,kk= _generate_nd_grid(grid_h, grid_w,grid_d)
    # convert shift to be compatible with the paper representation
    s = (-shift_size) % window_size
    offset = window_size / 2 - s
    ii = ii * window_size + offset
    jj = jj * window_size + offset
    kk = kk * window_size + offset


    input_coords = torch.stack([i.flatten(), j.flatten(),k.flatten()], 1).float()
    anchors_coords = torch.stack([ii.flatten(), jj.flatten(),jj.flatten()], 1).float()

    anchor_id = torch.cdist(input_coords, anchors_coords, p=2).argmin(1)
    mask = anchor_id[:, None] == anchor_id[None, :]
    
    beg_H,end_H=get_init_last(mask_other_shape[0],mask.shape[0])
    beg_W,end_W=get_init_last(mask_other_shape[1],mask.shape[1])
    mask=mask[beg_H:mask.shape[0]-end_H,beg_W:mask.shape[1]-end_W ]
    print(f"mmmm {mask.shape} mask_other_shape {mask_other_shape}")

    return mask


def local_nd_distance(*sizes, p=2.0, weights=None):
    if weights is None:
        weights = (1,) * len(sizes)
    assert len(sizes) == len(weights)
    grid = _generate_nd_grid(*sizes)
    grid = [i.flatten() * w for i, w in zip(grid, weights)]
    grid = torch.stack(grid, dim=1).float()
    d = torch.cdist(grid, grid, p=p)
    return d

def local_nd_pattern(*sizes, distance, p=2.0):
    d = local_nd_distance(*sizes, p=p)
    # print(d)
    return d < distance


def get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer):
    p= patch_size
    img_size=img_size
    res=[batch_size,embed_dim,img_size[0]/p[0],img_size[1]/p[1],img_size[2]/p[2]]
    for i in range(i_layer):
        res[1]=res[1]*2
        res[2]=int(res[2]//2)
        res[3]=int(res[3]//2)
        res[4]=int(res[4]//2)
    return res



def get_group_or_create(f,name):
    main_keys=np.array(list(f.keys()))
    main_group=[]
    if(name not in main_keys):
        main_group = f.create_group(name)
    else:
        main_group = f[name]
    return main_group   

def save_sputnik_sparse_tensor(top_group,name,sparse_tensor):
    main_keys=np.array(list(top_group.keys()))
    if(name not in main_keys):
        group=top_group.create_group(name)
        group.create_dataset(name="shape", data=np.array(list(sparse_tensor.shape)))
        group.create_dataset(name="values",data= sparse_tensor.values.detach().cpu().numpy())
        group.create_dataset(name="row_offsets", data=sparse_tensor.row_offsets.detach().cpu().numpy())
        group.create_dataset(name="column_indices", data=sparse_tensor.column_indices.detach().cpu().numpy())
        group.create_dataset(name="row_indices", data=sparse_tensor.row_indices.detach().cpu().numpy())
        for index,tensorr in enumerate(sparse_tensor._transp_info):
            group.create_dataset(name=f"{index}_transp_info",data= tensorr.detach().cpu().numpy())
            
def local_nd_distance_with_spacing(*sizes, p=2.0, weights=None,spacing=(1.0,1.0,1.0)):
    if weights is None:
        weights = (1,) * len(sizes)
    assert len(sizes) == len(weights)
    grid = _generate_nd_grid(*sizes)
    grid = [i.flatten() * w for i, w in zip(grid, weights)]
    grid = torch.stack(grid, dim=1).float()
    grid[:,0]=grid[:,0]*spacing[0]
    grid[:,1]=grid[:,1]*spacing[1]
    grid[:,2]=grid[:,2]*spacing[2]

    d = torch.cdist(grid, grid, p=p)
    return d              

def local_nd_pattern_with_spacing(*sizes, distance, p=2.0,spacing=(1.0,1.0,1.0)):
    d = local_nd_distance_with_spacing(*sizes, p=p,spacing=spacing)
    # print(d)
    return d < distance

def save_dist(loc_dists,to_save_dense_dists,im_size_group,loc_dist_name,distance):
    # gg=im_size_group.create_group(loc_dist_name)
    # if(to_save_dense_dists):
    #     loc_dists= einops.rearrange(loc_dists,'a b -> 1 a b')
    #     gg.create_dataset(name=f"dist_dense",data= loc_dists.detach().cpu().numpy())
    # else:
    #     # print(f"fffffffff {loc_dists.shape}  eye {torch.eye(loc_dists.shape[0]).shape}")
    #     loc_dists= einops.rearrange(loc_dists,'a b -> 1 a b')
    #     local_mask = loc_dists>distance
    #     indicies=torch.argwhere(local_mask)
    #     indicies=einops.rearrange(indicies,'a b -> b a')
    #     sparse_loc_dist= torch.sparse_coo_tensor(indicies, values=loc_dists[local_mask])
    #     gg.create_dataset(name=f"dist_sparse_indicies",data= sparse_loc_dist.coalesce().indices().detach().cpu().numpy())
    #     gg.create_dataset(name=f"dist_sparse_values",data= sparse_loc_dist.coalesce().values().detach().cpu().numpy())
    pass

def save_sparse_masks(distances,window_size,spacing,num_layers,patch_size,img_size,batch_size,embed_dim,f,to_save_dense_dists_all):
    
    for i_layer in range(num_layers):
        img_size_curr=get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer)
        distance=distances[i_layer]
        to_save_dense_dists=to_save_dense_dists_all[i_layer]
        img_size_curr=(img_size_curr[2],img_size_curr[3],img_size_curr[4])

        #top organization is based on the image size
        im_size_name=f"{int(img_size_curr[0])}_{int(img_size_curr[1])}_{int(img_size_curr[2])}"
        im_size_group=get_group_or_create(f,im_size_name)


        ## saving isovolumetric 
        local_mask=local_nd_pattern(img_size_curr[0],img_size_curr[1],img_size_curr[2],distance=distance)
        local_mask = SparseCS(local_mask, torch.device("cpu"))
        distance_group=get_group_or_create(im_size_group,f"dist_{distance}")
        save_sputnik_sparse_tensor(distance_group,"iso_vol",local_mask)
        loc_dists=local_nd_distance(img_size_curr[0],img_size_curr[1],img_size_curr[2])
        loc_dist_name="iso_dist"
        save_dist(loc_dists,to_save_dense_dists,distance_group,loc_dist_name,distance)
        ## saving non isovolumetric
        local_mask=local_nd_pattern_with_spacing(img_size_curr[0],img_size_curr[1],img_size_curr[2],distance=distance,spacing=spacing)
        local_mask_shape= local_mask.shape
        local_mask = SparseCS(local_mask, torch.device("cpu"))
        distance_spacing_group=get_group_or_create(im_size_group,f"dist_{distance}_spacing_{spacing[0]}_{spacing[1]}_{spacing[2]}")
        save_sputnik_sparse_tensor(distance_spacing_group,"non_iso_vol",local_mask)
        loc_dists=local_nd_distance_with_spacing(img_size_curr[0],img_size_curr[1],img_size_curr[2],spacing=spacing)
        loc_dist_name="non_iso_dist"
        save_dist(loc_dists,to_save_dense_dists,distance_spacing_group,loc_dist_name,distance)


        ###saving swin
        
        swin_group=get_group_or_create(im_size_group,"swin")
        #set on the base of window size
        window_size_group=get_group_or_create(swin_group,f"window_{window_size}")
        swin_mask=swin_attention_pattern_3D(img_size_curr[0], img_size_curr[1],img_size_curr[2], window_size, shift_size=2,mask_other_shape=local_mask_shape)
        
        swin_mask = SparseCS(swin_mask, torch.device("cpu"))
        save_sputnik_sparse_tensor(window_size_group,"main",swin_mask)



f = h5py.File(h5_path,'w')
feature_size=64
embed_dim=feature_size
# save_sparse_masks((4,4,4,4),4,(3.299999952316284,0.78125, 0.78125),4,(2,2,2),(48, 192, 160),1,embed_dim,f)
# save_sparse_masks((8,8,8,8),6,(3.299999952316284,0.78125, 0.78125),4,(2,2,2),(48, 192, 160),1,embed_dim,f)
save_sparse_masks((8,8,8,8,8),6,(3.299999952316284,0.78125, 0.78125),4,(2,2,2),(32, 32, 32),1,embed_dim,f,(False,False,False,False))
# save_sparse_masks((8,8,8,8),6,(3.299999952316284,0.78125, 0.78125),3,(2,2,2),(32, 32, 32),1,embed_dim,f,(False,False,True))


# save_sparse_masks((4,8,16,32),6,(3.299999952316284,0.78125, 0.78125),4,(2,2,2),(48, 192, 160),1,embed_dim,f)
# # save_sparse_masks((4,8,16,16),6,(3.299999952316284,0.78125, 0.78125),4,(2,2,2),(48, 192, 160),1,embed_dim,f)
# save_sparse_masks((8,16,32,64),6,(3.299999952316284,0.78125, 0.78125),4,(2,2,2),(48, 192, 160),1,embed_dim,f)

f.close()