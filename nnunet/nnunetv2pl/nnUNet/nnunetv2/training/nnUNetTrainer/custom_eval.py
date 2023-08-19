from comet_ml import Experiment
import comet_ml
import SimpleITK as sitk
import glob
import monai
import os
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
import functools
import multiprocessing as mp
import os
from functools import partial
import numpy as np
import pandas as pd
from toolz.itertoolz import groupby
import SimpleITK as sitk
import torch
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from torch import autocast, nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context


def analyze_single_label(uniq_num,centers, big_mask, connected,in_min):
    infered_inner=(connected==uniq_num)
    total=np.sum(infered_inner.flatten())
    inn= np.sum(np.logical_and(infered_inner,big_mask).flatten())/total
    # cov= np.sum(np.logical_and(infered,centers).flatten())/np.sum(centers.flatten())
    res= (inn>in_min) #and (cov>cover_min)
    return res

def get_my_specifity(bi,inn,twos,curr,bigger_mask):
    centers= twos[bi,:,:,:]
    inferred=curr[bi,:,:,:]
    big_mask=bigger_mask[bi,:,:,:]

    if(np.sum(inferred.flatten())==0):
        return 1.0

    connected=sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(inferred.astype(int))))
    uniqq=np.unique(connected)
    uniqq= list(filter(lambda el:el>0,uniqq))
    in_min=0.5
    res= list(map(lambda uniq_num: analyze_single_label(uniq_num,centers, big_mask, connected,in_min), uniqq))
    res= np.mean(np.array(res).astype(int))
    return res

def is_sth_in_areas(uniq_num,arr,inferred):

    bool_arr=(arr.copy()==uniq_num)
    summ=np.sum(inferred[bool_arr].flatten())
    res= summ>0
    return res

def get_connected_components_num(arr):
    connected=sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(arr.astype(int))))
    uniqq=np.unique(connected)
    uniqq= list(filter(lambda el:el>0,uniqq))
    return len(uniqq)    

def get_my_sensitivity(bi,inn,twos,curr,bigger_mask):
    curr_in= inn[bi,:,:,:]
    curr_twos= twos[bi,:,:,:]
    inferred=curr[bi,:,:,:]
    curr_bigger_mask=bigger_mask[bi,:,:,:]
    # if(epoch%10==0):
    #     curr_num=bi*100+batch_id
    #     sitk.WriteImage(sitk.GetImageFromArray(inferred.astype(np.uint8)), f"{folder_path}/{curr_num}_inferred.nii.gz")
    #     sitk.WriteImage(sitk.GetImageFromArray(curr_bigger_mask.astype(np.uint8)), f"{folder_path}/{curr_num}_big_mask.nii.gz")
    #     sitk.WriteImage(sitk.GetImageFromArray(curr_twos.astype(np.uint8)), f"{folder_path}/{curr_num}_centers.nii.gz")
        

        
    
    total = np.sum(curr_in.flatten())
    total_twos = np.sum(curr_twos.flatten())
    curr_percent_in=np.zeros(1)
    curr_percent_covered=np.zeros(1)

    if(total_twos==0):
        return -1.0
    if(total==0):
        return 0.0
    
    components_ones = get_connected_components_num(curr_bigger_mask)
    components_twos = get_connected_components_num(curr_twos)

    arr=curr_bigger_mask
    if(components_twos>components_ones):
        # print(f"ones are mergeddd")
        arr=curr_twos

    connected=sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(arr.astype(int))))
    uniqq=np.unique(connected)
    uniqq= list(filter(lambda el:el>0,uniqq))


    res=np.array(list(map(lambda uniq_num: is_sth_in_areas(uniq_num,connected,inferred),uniqq)))
    res=res.astype(int)
    return np.mean(res.flatten())

    
def calc_custom_metrics(group_name,f):    
    batch_nums= f[group_name].keys()

    target=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/target"][:,:,:,:], batch_nums))
    predicted_segmentation_onehot=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/predicted_segmentation_onehot"][:,:,:,:], batch_nums))

    target=np.concatenate(target,axis=0)
    predicted_segmentation_onehot=np.concatenate(predicted_segmentation_onehot,axis=0)

    return calc_custom_metrics_inner(target,predicted_segmentation_onehot)


def calc_custom_metrics_inner(target,predicted_segmentation_onehot):


    curr=predicted_segmentation_onehot
    bigger_mask= (target>0)[:,0,:,:,:]
    # curr= torch.sum(curr,dim=1)
    inn = curr & bigger_mask
    twos= (target==2)[:,0,:,:,:]
    shapp= target.shape

    
    with mp.Pool(processes = mp.cpu_count()) as pool:
        my_sensitivity=pool.map(partial(get_my_sensitivity,inn=inn,twos=twos,curr=curr,bigger_mask=bigger_mask),range(shapp[0]))
        my_specificity=pool.map(partial(get_my_specifity,inn=inn,twos=twos,curr=curr,bigger_mask=bigger_mask),range(shapp[0]))
    
    # with mp.Pool(processes = mp.cpu_count()) as pool:
        
    
    my_sensitivity=list(map(partial(get_my_sensitivity,inn=inn,twos=twos,curr=curr,bigger_mask=bigger_mask),range(shapp[0])))
    my_specificity=list(map(partial(get_my_specifity,inn=inn,twos=twos,curr=curr,bigger_mask=bigger_mask),range(shapp[0])))
    

    my_sensitivity=list(filter(lambda el: np.array(el).flatten()[0]>-1,my_sensitivity  ))      
    if(len(my_sensitivity)>0):
        my_sensitivity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_sensitivity))))
        my_specificity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_specificity))))
        is_correct= (my_sensitivity*2+my_specificity)/3
    
    total = np.sum(curr.flatten())
    
    if(total>0):
        percent_in=np.sum(inn.flatten())/(total)
    if(total>0):
        out = (curr) & (~bigger_mask)
        percent_out= np.sum(out.flatten())/total

    
    two_sum=np.sum(twos.flatten())
    if(two_sum>0):
        percent_covered=  np.sum(((curr) & (twos)).flatten())/two_sum
    
    # print(f"tttttt is_correct {is_correct} total {total} ((curr) & (twos)).sum() {((curr) & (twos)).sum()}  (curr) & (~bigger_mask) {((curr) & (~bigger_mask)).sum()}")
    is_correct=np.array(is_correct).flatten()

    my_sensitivity=np.array(np.nanmean(np.array(my_sensitivity).flatten()))
    my_specificity=np.array(np.nanmean(np.array(my_specificity).flatten()))

    percent_in=np.array([percent_in]).flatten()
    percent_out=np.array([percent_out]).flatten()
    percent_covered=np.array([percent_covered]).flatten()    

    return np.array([np.nanmean(percent_in).flatten()
                     ,np.nanmean(percent_out).flatten()
                     ,np.nanmean(percent_covered).flatten()
                     ,np.nanmean(is_correct).flatten()
                     ,np.nanmean(my_sensitivity).flatten()
                     ,np.nanmean(my_specificity).flatten()])





def save_to_hdf5(f,inner_id,group_name,batch_id,target,output,data):
    output=torch.stack([output[:,0,:,:,:],output[:,2,:,:,:]],dim=1)
    output_seg = output.argmax(1)[:, None]
    predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
    predicted_segmentation_onehot.scatter_(1, output_seg, 1)
    del output_seg   
    curr=predicted_segmentation_onehot.round().bool()[:,1,:,:,:]  
    target_str= f"{group_name}/{batch_id}/target"
    predicted_segmentation_onehot_str= f"{group_name}/{batch_id}/predicted_segmentation_onehot"
    data_str= f"{group_name}/{batch_id}/data"
    if(group_name not in f.keys()):
        f.create_group(group_name)
    if(f"{batch_id}" not in f[group_name].keys() ):
        f.create_dataset(target_str,data= target.detach().cpu().numpy())
        f.create_dataset(predicted_segmentation_onehot_str,data= curr.detach().cpu().numpy())
        f.create_dataset(data_str,data= data.detach().cpu().numpy())
    else:
        f[target_str] = target.detach().cpu().numpy()
        f[predicted_segmentation_onehot_str]= curr.detach().cpu().numpy()
        f[data_str] = data.detach().cpu().numpy()


def save_for_metrics(epoch,target,output,data,log_every_n,batch_id,f,group_name):
    if(isinstance(output, list)):
        # list(map(lambda i: save_to_hdf5(f,i,group_name,batch_id,target[i],output[i],data[i]),range(len(output))))
        save_to_hdf5(f,0,group_name,batch_id,target[0],output[0],data[0])
    else:
        save_to_hdf5(f,0,group_name,batch_id,target,output,data)  


