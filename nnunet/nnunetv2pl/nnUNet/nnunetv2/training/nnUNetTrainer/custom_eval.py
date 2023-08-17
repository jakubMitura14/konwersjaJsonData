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

def get_my_specifity(bi,inn,twos,curr,epoch,folder_path,batch_id,bigger_mask):
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

def get_my_sensitivity(bi,inn,twos,curr,epoch,folder_path,batch_id,bigger_mask):
    curr_in= inn[bi,:,:,:]
    curr_twos= twos[bi,:,:,:]
    inferred=curr[bi,:,:,:]
    curr_bigger_mask=bigger_mask[bi,:,:,:]
    if(epoch%10==0):
        curr_num=bi*100+batch_id
        sitk.WriteImage(sitk.GetImageFromArray(inferred.astype(np.uint8)), f"{folder_path}/{curr_num}_inferred.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(curr_bigger_mask.astype(np.uint8)), f"{folder_path}/{curr_num}_big_mask.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(curr_twos.astype(np.uint8)), f"{folder_path}/{curr_num}_centers.nii.gz")
        
    
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
        print(f"ones are mergeddd")
        arr=curr_twos

    connected=sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(arr.astype(int))))
    uniqq=np.unique(connected)
    uniqq= list(filter(lambda el:el>0,uniqq))


    res=np.array(list(map(lambda uniq_num: is_sth_in_areas(uniq_num,connected,inferred),uniqq)))
    res=res.astype(int)
    return np.mean(res.flatten())

def calc_custom_metrics(epoch,target,output ):
    percent_in= np.zeros(1)
    percent_out=np.zeros(1)
    percent_covered=np.zeros(1)
    is_correct=np.zeros(1)
    my_sensitivity=np.zeros(1)
    my_specificity=np.zeros(1)


    if(epoch%10==0 and epoch>0 ): #and epoch>0

        #important ! we are purposfully ignoring the channel 1 in argmax !
        output=torch.stack([output[:,0,:,:,:],output[:,2,:,:,:]],dim=1)
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        bigger_mask= (target>0)[:,0,:,:,:]
        curr=predicted_segmentation_onehot.round().bool()[:,1,:,:,:]
        
        # curr= torch.sum(curr,dim=1)
        inn = curr & bigger_mask
        twos= (target==2)[:,0,:,:,:]
        shapp= target.shape




        inn=inn.detach().cpu().numpy()
        twos=twos.detach().cpu().numpy()
        curr=curr.detach().cpu().numpy()
        bigger_mask=bigger_mask.detach().cpu().numpy()
        # curr= np.round(np.random.random(curr.shape)).astype(bool) #TODO remove            

            
        
        base='/workspaces/konwersjaJsonData/explore/validation_to_look_into'
        folder_path=f"{base}/{epoch}"
        os.makedirs(folder_path,exist_ok=True)
        
        with mp.Pool(processes = mp.cpu_count()) as pool:
            my_sensitivity=pool.map(partial(get_my_sensitivity,inn=inn,twos=twos,curr=curr,epoch=epoch,folder_path=folder_path,batch_id=batch_id,bigger_mask=bigger_mask),range(shapp[0]))
        
        with mp.Pool(processes = mp.cpu_count()) as pool:
            my_specificity=pool.map(partial(get_my_specifity,inn=inn,twos=twos,curr=curr,epoch=epoch,folder_path=folder_path,batch_id=batch_id,bigger_mask=bigger_mask),range(shapp[0]))
        

        my_sensitivity=list(filter(lambda el: np.array(el).flatten()[0]>-1,my_sensitivity  ))      
        if(len(my_sensitivity)>0):
            my_sensitivity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_sensitivity))))
            my_specificity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_specificity))))
            is_correct= (my_sensitivity+my_specificity)/2
        
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

        return percent_in,percent_out,percent_covered,is_correct,my_sensitivity,my_specificity




def my_validation_step(batch: dict,device,network,loss,label_manager,epoch) -> dict:
    data = batch['data']
    target = batch['target']

    data = data.to(device, non_blocking=True)
    if isinstance(target, list):
        target = [i.to(device, non_blocking=True) for i in target]
    else:
        target = target.to(device, non_blocking=True)

    # Autocast is a little bitch.
    # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    # So autocast will only be active if we have a cuda device.
    with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
        output = network(data)
        del data
        l = loss(output, target)

    # we only need the output with the highest output resolution
    output = output[0]
    target = target[0]

    # the following is needed for online evaluation. Fake dice (green line)
    axes = [0] + list(range(2, len(output.shape)))

    if label_manager.has_regions:
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
    else:
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

    if label_manager.has_ignore_label:
        if not label_manager.has_regions:
            mask = (target != label_manager.ignore_label).float()
            # CAREFUL that you don't rely on target after this line!
            target[target == label_manager.ignore_label] = 0
        else:
            mask = 1 - target[:, -1:]
            # CAREFUL that you don't rely on target after this line!
            target = target[:, :-1]
    else:
        mask = None

    tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

    tp_hard = tp.detach().cpu().numpy()
    fp_hard = fp.detach().cpu().numpy()
    fn_hard = fn.detach().cpu().numpy()
    if not label_manager.has_regions:
        # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        # (softmax training) there needs tobe one output for the background. We are not interested in the
        # background Dice
        # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

    percent_in,percent_out,percent_covered,is_correct,my_sensitivity,my_specificity=calc_custom_metrics(epoch,target,output )

    return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard
                    , 'percent_in' :percent_in,'percent_out':percent_out,'percent_covered':percent_covered,'is_correct':is_correct
                    ,'my_sensitivity':my_sensitivity,'my_specificity':my_specificity   }