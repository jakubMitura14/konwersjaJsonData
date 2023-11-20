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
from torch import autocast, nn

import numpy as np
import scipy.stats as st
from batchgenerators.utilities.custom_types import ScalarType, sample_scalar
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from transformers import AutoProcessor
import einops
        # tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
from PIL import Image as im
import torchvision.transforms as T
import monai
import itertools
from more_itertools import batched


class My_priming_setter(LocalTransform):
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
        
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            dat_curr= data[bi,:,:,:,:]
            target_curr= target[bi,:,:,:]
            twos_curr= dat_curr[5,:,:,:]==2
            res_bool= np.zeros_like(twos_curr)
            indicies=np.argwhere(twos_curr)
            rng = np.random.default_rng()
            rng.shuffle(indicies,axis=0)
            n_points=1
            indicies=indicies[0:n_points,:]

            res_bool[indicies[:,0],indicies[:,1],indicies[:,2]]=True
            res_bool.astype(float)
            data[bi, 5,:,:,:] =res_bool.astype(float)
        data_dict["data"]=data      
        return data_dict
     
class My_gpu_pseudo_lesion_adder(nn.Module):

    def __init__(self, n,k,mean_0,mean_1,mean_2,mean_3,std_0,std_1,std_2,std_3,is_anatomic,mult_old_a,mult_old_b):
        super(My_gpu_pseudo_lesion_adder, self).__init__()
        self.n = n
        self.k = k
        self.mean_0=mean_0
        self.mean_1=mean_1
        self.mean_2=mean_2
        self.mean_3=mean_3
        self.std_0=std_0
        self.std_1=std_1
        self.std_2=std_2
        self.std_3=std_3
        self.mult_old_a=mult_old_a
        self.mult_old_b=mult_old_b
        self.is_anatomic=is_anatomic

        self.blur=monai.transforms.GaussianSmooth()#T.GaussianBlur(kernel_size=(3,3,3), sigma=(0.01,0.02 ))

    def inner_dilatate(self,arr):
        arr= self.blur(arr.bool().float())
        return (arr>0.07)

    def my_dilatate(self,arr,n):
        arr=arr
        for i in range(n-1):
            arr=self.inner_dilatate(arr)
        return arr

    def my_shuffle(self,x,d):
        """
        taken from
        """
        # d = 1
        return x.take_along_dim(torch.sort(torch.rand(*x.shape))[1], d)

    def standardd(self,arr):
        minn= torch.min(arr)
        arr=(arr-minn)
        maxx= torch.max(arr)
        arr=arr/maxx
        arr=torch.nan_to_num(arr)
        return arr

    def forward(self, data, target_curr):
        """
        adding in random spots (but not where already some labels are) lesions that are 
        either low or high on both adc and hbv (they should be high hbv low adc)
        function is not batched
        """
        is_cuda=True

        img_shape= (data.shape[2],data.shape[3],data.shape[4])
        data= data[0,:,:,:,:]
        target_curr= target_curr[0,:,:,:]
        #checked for lesions 
        gauss_vals=np.array([ [[self.mean_0,self.std_0],[self.mean_1,self.std_1] ]
                            ,[[self.mean_2,self.std_2],[self.mean_3,self.std_3] ] ])
        #n - controlling how big the pseudo lesion will be
        n=torch.randint(low=2,high=self.n,size=(1,)).item()
        #k - controlling the numebr of pseudo lesions
        k=torch.randint(low=0,high=self.k,size=(1,)).item() 
        #1) get the target and dilatate it n+1 times (if its sum is above 0)
        if(self.is_anatomic):
            anatomic= torch.logical_or(data[-1,:,:,:]>0,data[-2,:,:,:])
            target_big= self.my_dilatate(anatomic,n)
        else:    
            target_big= self.my_dilatate((target_curr>0),n) #ndimage.binary_dilation((target_curr>0),iterations=n)
        #2) get indicies where dilatated target is still zero and choose random k indicies from it
        target_big=torch.argwhere(torch.logical_not(target_big)).detach().cpu().numpy()
        rng = np.random.default_rng()
        rng.shuffle(target_big,axis=0)
        target_big=target_big[0:k,:]
        #3) create new zero bool array of original size 
        res_bool= np.zeros(img_shape).astype(bool)
        #4) in a neww array set chosen points to True
        res_bool[target_big[:,0],target_big[:,1],target_big[:,2]]=True
        res_bool= torch.tensor(res_bool)
        if(is_cuda):
            res_bool=res_bool.cuda()
        #5) perform binary dilatation n-1 times and save
        res_bool_a= self.my_dilatate(res_bool,n-1)
        #6) perform last dilatation - and find indicies that were added in last dilatation
        res_bool_b= self.my_dilatate(res_bool_a,1)
        diff= torch.argwhere(torch.logical_and(torch.logical_not(res_bool_a), res_bool_b)).detach().cpu().numpy()
        res_bool_b=res_bool_b.detach().cpu().numpy()
        #7) set randomly 60% of them to false so we will have more realistic border
        rng = np.random.default_rng()
        rng.shuffle(diff,axis=0)
        len= diff.shape[0]
        diff=diff[0:len,:]
        res_bool_b[diff[:,0],diff[:,1],diff[:,2]]=False
        res_bool_b= torch.tensor(res_bool_b)
        if(is_cuda):
            res_bool_b=res_bool_b.cuda()

        #8) we get resulting boolean array calling it res_bool
        res_bool=res_bool_b
        #9) we index image (both channels) with res bool and set it to 0
        data_a=data[0,:,:,:]
        data_b=data[1,:,:,:]



        # dat_curr=np.stack([data_a,data_b])

        #10) we create new float array of the size like image
        #11) we set it with either mean and variance typical for lesions on hbv or adc and we set the same values 
            # for both channels
        rng = np.random.default_rng()    
        rng.shuffle(gauss_vals,axis=0)
        gauss_vals= torch.tensor(gauss_vals)
        if(is_cuda):
            gauss_vals=gauss_vals.cuda()

        noise_adc=torch.normal(gauss_vals[0,0,0], gauss_vals[0,0,1], size=img_shape)
        noise_hbv=torch.normal(gauss_vals[0,1,0], gauss_vals[0,1,1], size=img_shape)
        if(is_cuda):
            noise_adc=noise_adc.cuda()
            noise_hbv=noise_hbv.cuda()

        #12) we set evrything outside of res_bool to false    
        noise_adc[torch.logical_not(res_bool)]=1.0
        noise_hbv[torch.logical_not(res_bool)]=1.0

        data_a[res_bool]=data_a[res_bool]*noise_adc[res_bool]
        data_b[res_bool]=data_b[res_bool]*noise_hbv[res_bool]

        # noise_adc= self.standardd(noise_adc)
        # noise_hbv= self.standardd(noise_hbv)

        # noise= torch.stack([noise_adc,noise_hbv])
        # data_a= self.standardd(self.standardd(data_a)+noise_adc)
        # data_b= self.standardd(self.standardd(data_b)+noise_hbv)
                
        #13)we add image (with zeroad indexes) to array we got in previous step
        res=torch.stack([data_a,data_b])
        # res=res+noise
        #we need to add remaining images - anatomy
        data_c=data[2:,:,:,:]
        # data_c=np.expand_dims(data_c,axis=0)
        res= torch.concatenate([res,data_c],axis=0)
        res= einops.rearrange(res,'x y z c->1 x y z c')

        # self.mark_non_differentiable(res)
        return res
    