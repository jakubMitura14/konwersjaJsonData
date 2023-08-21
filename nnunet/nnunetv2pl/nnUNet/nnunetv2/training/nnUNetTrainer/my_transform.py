import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.local_transforms import *
from batchgenerators.transforms.local_transforms import LocalTransform
from abc import ABC
from typing import Tuple

import numpy as np
import scipy.stats as st
from batchgenerators.utilities.custom_types import ScalarType, sample_scalar
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from transformers import AutoProcessor
import einops
        # tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))



def augment_two_channel(dat_curr,target_curr):
    """
    adding in random spots (but not where already some labels are) lesions that are 
    either low or high on both adc and hbv (they should be high hbv low adc)
    function is not batched
    """
    img_shape= (dat_curr.shape[1],dat_curr.shape[2],dat_curr.shape[3])
    #checked for lesions 
    # #mean adc is 1094.62154301315 stdev 481.980487471925
    #hbv mean 19.4776261112059 std 4.75021004238441
    #for prostate no lesions
    #adc mean 1169.06275422957  std 123.857216768034
    #hbv mean 13.8230222890104  std 3.02186134116448

    gauss_vals=np.array([ [[1094.62154301315,30.0],[13.8230222890104,3.02186134116448 ] ]
                         ,[[1169.06275422957,30.0],[19.4776261112059,4.7502100423844] ] ])
    #n - controlling how big the pseudo lesion will be
    n=np.random.randint(2,6)
    #k - controlling the numebr of pseudo lesions
    k=np.random.randint(0,50) 
    #1) get the target and dilatate it n+1 times (if its sum is above 0)
    target_big= ndimage.binary_dilation((target_curr>0),iterations=n)
    #2) get indicies where dilatated target is still zero and choose random k indicies from it
    target_big=np.argwhere(np.logical_not(target_big))
    rng = np.random.default_rng()
    rng.shuffle(target_big,axis=0)
    target_big=target_big[0:k,:]
    #3) create new zero bool array of original size 
    res_bool= np.zeros(img_shape).astype(bool)
    #4) in a neww array set chosen points to True
    res_bool[target_big[:,0],target_big[:,1],target_big[:,2]]=True
    #5) perform binary dilatation n-1 times and save
    res_bool_a= ndimage.binary_dilation(res_bool,iterations=n-1)
    #6) perform last dilatation - and find indicies that were added in last dilatation
    res_bool_b= ndimage.binary_dilation(res_bool_a,iterations=1)
    diff= np.argwhere(np.logical_and(np.logical_not(res_bool_a), res_bool_b))
    #7) set randomly 60% of them to false so we will have more realistic border
    rng = np.random.default_rng()
    rng.shuffle(diff,axis=0)
    len= diff.shape[0]
    diff=diff[0:len,:]
    res_bool_b[diff[:,0],diff[:,1],diff[:,2]]=False
    #8) we get resulting boolean array calling it res_bool
    res_bool=res_bool_b
    #9) we index image (both channels) with res bool and set it to 0
    data_a=dat_curr[0,:,:,:]
    data_b=dat_curr[1,:,:,:]
    data_a[res_bool]=0
    data_b[res_bool]=0
    # dat_curr=np.stack([data_a,data_b])

    #10) we create new float array of the size like image
    #11) we set it with either mean and variance typical for lesions on hbv or adc and we set the same values 
        # for both channels
    rng = np.random.default_rng()    
    rng.shuffle(gauss_vals,axis=0)
    noise_adc=np.random.normal(gauss_vals[0,0,0], gauss_vals[0,0,1], size=img_shape)
    noise_hbv=np.random.normal(gauss_vals[0,1,0], gauss_vals[0,1,1], size=img_shape)
    #12) we set evrything outside of res_bool to false    
    noise_adc[np.logical_not(res_bool)]=0.0
    noise_hbv[np.logical_not(res_bool)]=0.0
    noise= np.stack([noise_adc,noise_hbv])
    #13)we add image (with zeroad indexes) to array we got in previous step
    res=np.stack([data_a,data_b])
    res=res+noise
    #we need to add remaining images - anatomy
    data_c=dat_curr[2:,:,:,:]
    # data_c=np.expand_dims(data_c,axis=0)
    return np.concatenate([res,data_c],axis=0)


class My_PseudoLesion_adder(LocalTransform):
    def __init__(self,
                 max_strength: ScalarType = 1.,
                 same_for_all_channels: bool = True,
                 mean_centered: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 clip_intensities: bool = False,
                 data_key: str = "data"):
        super().__init__(1.0)


    def __call__(self, **data_dict):
        data = data_dict.get("data")
        target= data_dict.get("seg")
        # print(f"iiiiiiiiiin transform {data.shape}  keys {data_dict.keys()} ")
        
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            dat_curr= data[bi,:,:,:,:]
            target_curr= target[bi,:,:,:]
            dat_curr=augment_two_channel(dat_curr,target_curr)
            #)we save result in data dictionary
            data[bi, :,:,:,:] =dat_curr
        return data_dict


class One_former_preprocess(LocalTransform):
    def __init__(self):
        super().__init__(1.0)
        self.processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        
    def __call__(self, **data_dict):
        data = data_dict.get("data")
        target= data_dict.get("target")
        orig_shape=data.shape
        assert data is not None, "Could not find data key '%s'" % self.data_key
        print(f"ppppppppppppppp data {data.shape}")

        data = einops.rearrange(data,'b c x y z  -> (b z) c x y')
        target = einops.rearrange(target,'b c x y z -> (b z) c x y')

        data_a=data[11,0:3,:,:]#TODO change
        target=target[10:11,:,:,:]#TODO change
        # data = np.nan_to_num(data,0)
        data_a = (data_a-np.min(data_a.flatten()))/((np.max(data_a.flatten())-np.min(data_a.flatten()))+0.000000000001)

        data_b=data[12,0:3,:,:]#TODO change
        target=target[10:11,:,:,:]#TODO change
        # data = np.nan_to_num(data,0)
        data_b = (data_b-np.min(data_b.flatten()))/((np.max(data_b.flatten())-np.min(data_b.flatten()))+0.000000000001)

        data = self.processor(images=[data_a,data_b], task_inputs=["semantic"], return_tensors="pt")
        #data = einops.rearrange(data,'(b z) c x y->b c x y z',b=orig_shape[0])
        #target = einops.rearrange(target,'(b z) c x y c->b c x y z',b=orig_shape[0])      
        
        return {"data":data,"target":target,"orig_shape":orig_shape}

