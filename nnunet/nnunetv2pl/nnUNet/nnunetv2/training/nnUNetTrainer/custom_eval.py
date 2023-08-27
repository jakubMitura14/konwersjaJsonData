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
import shutil
import itertools
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer


def analyze_single_label(uniq_num,centers, big_mask, connected,in_min):
    infered_inner=(connected==uniq_num)
    total=np.sum(infered_inner.flatten())
    inn= np.sum(np.logical_and(infered_inner,big_mask).flatten())/total
    # cov= np.sum(np.logical_and(infered,centers).flatten())/np.sum(centers.flatten())
    res= (inn>in_min) #and (cov>cover_min)
    return res

def get_my_specifity(arrs):

    curr_in,centers,inferred,big_mask,data=arrs
    
    if(np.sum(inferred.flatten())==0):
        return 1.0, 0

    connected=sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(inferred.astype(int))))
    uniqq=np.unique(connected)
    uniqq= list(filter(lambda el:el>0,uniqq))
    in_min=0.5
    res= list(map(lambda uniq_num: analyze_single_label(uniq_num,centers, big_mask, connected,in_min), uniqq))
    res= np.mean(np.array(res).astype(int))
    return res,len(uniqq)

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

def get_my_sensitivity(arrs):
    curr_in,curr_twos,inferred,curr_bigger_mask,data=arrs
    
    # curr_in= inn#[bi,:,:,:]
    # curr_twos= twos#[bi,:,:,:]
    # inferred=curr#[bi,:,:,:]
    # curr_bigger_mask=bigger_mask#[bi,:,:,:]
    # print(f"get_my_sensitivity {curr_in.shape  }")

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

def save_single_arr(image_array,batch_idd, bn, c,for_explore,name,typee ):
    # image_array,batch_idd, bn, c,for_explore,name,typee=args
    # im= sitk.GetImageFromArray(image_array)
    folder=f"{for_explore}/{int(batch_idd)*100+int(bn)}" #/{name}_{c}
    Path(folder).mkdir( parents=True, exist_ok=True )
    path=f"{folder}/im_{name}_{c}.nii.gz"    
    # print(f"ffffffffff folder {folder} sh {image_array.shape}")
    sitk.WriteImage(sitk.GetImageFromArray(image_array.astype(typee)), path)
    return path


def get_sensitivity_and_specificity(arrs_tupl,for_explore,batch_idd,to_save_files):
    bn,arrs=arrs_tupl
    ress=get_my_specifity(arrs)
    specificity=ress[0]
    num_components=ress[1]
    sensitivity=get_my_sensitivity(arrs)
    # print(f"specificity {specificity} sensitivity {sensitivity}")
    curr_in,curr_twos,inferred,curr_bigger_mask,data =arrs
    #if(False):
    if(to_save_files):
        
        save_single_arr(curr_in,batch_idd, bn, 0,for_explore,"curr_in",np.uint8 )
        save_single_arr(curr_twos,batch_idd, bn, 0,for_explore,"curr_twos",np.uint8 )
        save_single_arr(inferred,batch_idd, bn, 0,for_explore,"inferred",np.uint8 )
        save_single_arr(curr_bigger_mask,batch_idd, bn, 0,for_explore,"curr_bigger_mask",np.uint8 )

        save_single_arr(data[0,:,:,:],batch_idd, bn, 0,for_explore,"data",float)
        save_single_arr(data[1,:,:,:],batch_idd, bn, 1,for_explore,"data",float)
        save_single_arr(data[2,:,:,:],batch_idd, bn, 2,for_explore,"data",float)
        save_single_arr(data[3,:,:,:],batch_idd, bn, 3,for_explore,"data",float)
        save_single_arr(data[4,:,:,:],batch_idd, bn, 4,for_explore,"data",float)
        
    return specificity,sensitivity,num_components,np.sum(inferred.flatten())


# def save_files(arrs_tupl,for_explore,batch_idd,to_save_files):
#     bn,arrs=arrs_tupl
#     curr_in,curr_twos,inferred,curr_bigger_mask,data =arrs
#     #if(False):
#     if(to_save_files):
        
#         save_single_arr(curr_in,batch_idd, bn, 0,for_explore,"curr_in",np.uint8 )
#         save_single_arr(curr_twos,batch_idd, bn, 0,for_explore,"curr_twos",np.uint8 )
#         save_single_arr(inferred,batch_idd, bn, 0,for_explore,"inferred",np.uint8 )
#         save_single_arr(curr_bigger_mask,batch_idd, bn, 0,for_explore,"curr_bigger_mask",np.uint8 )

#         save_single_arr(data[0,:,:,:],batch_idd, bn, 0,for_explore,"data",float)
#         save_single_arr(data[1,:,:,:],batch_idd, bn, 1,for_explore,"data",float)
#         save_single_arr(data[2,:,:,:],batch_idd, bn, 2,for_explore,"data",float)
#         save_single_arr(data[3,:,:,:],batch_idd, bn, 3,for_explore,"data",float)
#         save_single_arr(data[4,:,:,:],batch_idd, bn, 4,for_explore,"data",float)
        


# def save_batched_to_file(for_explore,batch_ids,name,arr,typee):
#     batch_idd=batch_ids[0]
#     b_num_local=arr.shape[0]
#     c_num_local=arr.shape[1]
    
#     listed=list(map(lambda bn: 
#                     list(map(lambda c: (arr[bn,c,:,:,:],batch_idd, bn, c ,for_explore,name,typee) ,range(c_num_local) )) 
#                     , range(b_num_local)))
#     listed=itertools.chain(*listed)
#     with mp.Pool(processes = mp.cpu_count()) as pool:
#         pool.map(save_single_arr,listed)
    
#     return listed
    







def concat_local(batch_ids,f,group_name,name):
    res=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/{name}"][:,:,:,:],batch_ids))
    return np.concatenate(res,axis=0)
 
def concat_local_data(batch_ids,f,group_name,name):
    res=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/{name}"][:,:,:,:,:],batch_ids))
    return np.concatenate(res,axis=0)
    




def calc_custom_metrics(group_name,f,for_explore,to_save_files,anatomy_metr=False):    
    batch_nums= np.array(list(f[group_name].keys()))
    if(batch_nums.shape[0]<3):
        batch_nums=np.array_split(batch_nums, 1)
    else:
        batch_nums=np.array_split(batch_nums, 3)    
    # target=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/target"][:,:,:,:], batch_nums))
    # predicted_segmentation_onehot=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/predicted_segmentation_onehot"][:,:,:,:], batch_nums))

    # target=np.concatenate(target,axis=0)
    # predicted_segmentation_onehot=np.concatenate(predicted_segmentation_onehot,axis=0)
    if(to_save_files):
        os.makedirs(for_explore,exist_ok=True)
        shutil.rmtree(for_explore)   
        os.makedirs(for_explore,exist_ok=True)


    res= list(map(lambda batch_ids: calc_custom_metrics_inner(concat_local_data(batch_ids,f,group_name,"target")
                                                              ,concat_local(batch_ids,f,group_name,"predicted_segmentation_onehot")
                                                              ,concat_local_data(batch_ids,f,group_name,"data")
                                                              ,f,for_explore,to_save_files,batch_ids,anatomy_metr=anatomy_metr),batch_nums))

    if(anatomy_metr):
        res= itertools.chain(*res)
        metrics_names= np.unique(np.array(list(map(lambda tupl:tupl[0] ,res))))
        filtered=list(map(lambda name: list(filter(lambda tupl: tupl[0]==name ,res ))  , metrics_names))
        filtered= list(map( lambda listt: np.mean(np.array(list(map(lambda tupl :tupl[1] ,listt )))) ,filtered))
        return list(zip(metrics_names,filtered))
    res=np.concatenate(res,axis=-1)
    # res= list(map(lambda batch_ids: calc_custom_metrics_inner(f[f"{group_name}/{batch_id}/target"][:,:,:,:],f[f"{group_name}/{batch_id}/predicted_segmentation_onehot"][:,:,:,:]),batch_nums))
    res= np.mean(res,axis=-1)
    return res

def prep_arr_list(inn,twos,curr,bigger_mask,data,batch_num):
    
    return list(map(lambda bi: (inn[bi,:,:,:],twos[bi,:,:,:],curr[bi,:,:,:],bigger_mask[bi,:,:,:],data[bi,:,:,:,:]  ) ,range(batch_num)))


def save_arrs_anatomy(bn,predicted_segmentation_onehot,data,target,batch_idd,for_explore):
    save_single_arr(predicted_segmentation_onehot,batch_idd[bn,:,:,:], bn, 0,for_explore,"predicted_segmentation_onehot",np.uint8 )
    save_single_arr(target[bn,1,:,:,:],batch_idd, bn, 0,for_explore,"target",np.uint8 )
    save_single_arr(data[bn,1,:,:,:],batch_idd, bn, 0,for_explore,"t2w",float )


def calc_custom_metrics_inner(target,predicted_segmentation_onehot,data,f,for_explore,to_save_files,batch_ids,anatomy_metr):
    percent_in= np.zeros(1)
    percent_out=np.zeros(1)
    percent_covered=np.zeros(1)
    is_correct=np.zeros(1)
    my_sensitivity=np.zeros(1)
    my_specificity=np.zeros(1)
    shapp= target.shape
    batch_idd=int(batch_ids[0])




    ####### full anatomy metrics
    if(anatomy_metr):
        arrs=list(map(lambda bi: (predicted_segmentation_onehot[bi,:,:,:],target[bi,:,:,:]) ,range(shapp[0])))
        metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'), metric.VolumeSimilarity()]
        labels = {1: 'segmentation' }
        evaluator = eval_.SegmentationEvaluator(metrics, labels)      
        metr_res ='/workspaces/konwersjaJsonData/explore/metr.csv'
        # batch_idd=int(batch_ids[0])
        list(map(lambda bi:evaluator.evaluate(sitk.GetImageFromArray(predicted_segmentation_onehot[bi,:,:,:])
                                                , sitk.GetImageFromArray(target[bi,:,:,:])
                                                , (batch_idd*100)+bi) ,range(shapp[0])))        
        functions = {'MEAN': np.mean}
        writer.CSVStatisticsWriter(metr_res, functions=functions).write(evaluator.results)
        frame = pd.read_csv(metr_res,header=0,sep=";")
        rows = frame.iterrows()
        rows= list(map(lambda roww: (roww[1]['METRIC'],roww[1]['VALUE']),rows))
        list(map(lambda bn:save_arrs_anatomy(bn,predicted_segmentation_onehot,data,target,batch_idd,for_explore),range(shapp[0]) ))

        

        return rows
    ####### lesions metrics


    curr=predicted_segmentation_onehot
    bigger_mask= (target>0)[:,0,:,:,:]
    # curr= torch.sum(curr,dim=1)
    inn = curr & bigger_mask
    twos= (target==2)[:,0,:,:,:]
    
    
    total = np.sum(curr.flatten())
    
    if(total>0):
        percent_in=np.sum(inn.flatten())/(total)
    if(total>0):
        out = (curr) & (~bigger_mask)
        percent_out= np.sum(out.flatten())/total

    
    two_sum=np.sum(twos.flatten())
    if(two_sum>0):
        percent_covered=  np.sum(((curr) & (twos)).flatten())/two_sum
    
    
    
    percent_in=np.array([percent_in]).flatten()
    percent_out=np.array([percent_out]).flatten()
    percent_covered=np.array([percent_covered]).flatten()    
    

    arrs=prep_arr_list(inn,twos,curr,bigger_mask,data,shapp[0])
    
    del inn
    del twos
    del curr
    del bigger_mask

    ress=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        ress=pool.map(partial(get_sensitivity_and_specificity,for_explore=for_explore,batch_idd=batch_idd,to_save_files=to_save_files),enumerate(arrs))
    
    ress=np.array(ress)
    my_specificity=ress[:,0]
    my_sensitivity=ress[:,1]
    num_components=ress[:,2]
    in_inferred=ress[:,3]
    
    # list(map(lambda arrs_tupl :save_files(arrs_tupl,for_explore,batch_idd,to_save_files),enumerate(arrs)))
    
    # my_specificity,my_sensitivity,num_components,in_inferred=list(map(partial(get_sensitivity_and_specificity,for_explore=for_explore,batch_idd=batch_idd,to_save_files=to_save_files),enumerate(arrs)))
        # my_sensitivity=pool.map(get_my_sensitivity,arrs)
        # my_specificity=pool.map(get_my_specifity,arrs)
    # with mp.Pool(processes = mp.cpu_count()) as pool:
        
    
    # my_sensitivity=list(map(partial(get_my_sensitivity,inn=inn,twos=twos,curr=curr,bigger_mask=bigger_mask),range(shapp[0])))
    # my_specificity=list(map(partial(get_my_specifity,inn=inn,twos=twos,curr=curr,bigger_mask=bigger_mask),range(shapp[0])))
    

    my_sensitivity=list(filter(lambda el: np.array(el).flatten()[0]>-1,my_sensitivity  ))      
    if(len(my_sensitivity)>0):
        my_sensitivity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_sensitivity))))
        my_specificity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_specificity))))
        num_components= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),num_components))))
        in_inferred= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),in_inferred))))
        is_correct= (my_sensitivity*2+my_specificity)/3
    
  
    # print(f"tttttt is_correct {is_correct} total {total} ((curr) & (twos)).sum() {((curr) & (twos)).sum()}  (curr) & (~bigger_mask) {((curr) & (~bigger_mask)).sum()}")
    is_correct=np.array(is_correct).flatten()

    my_sensitivity=np.array(np.nanmean(np.array(my_sensitivity).flatten()))
    my_specificity=np.array(np.nanmean(np.array(my_specificity).flatten()))
    
    num_components=np.array(np.nanmean(np.array(num_components).flatten()))
    in_inferred=np.array(np.nanmean(np.array(in_inferred).flatten()))


    return np.array([np.nanmean(percent_in).flatten()
                     ,np.nanmean(percent_out).flatten()
                     ,np.nanmean(percent_covered).flatten()
                     ,np.nanmean(is_correct).flatten()
                     ,np.nanmean(my_sensitivity).flatten()
                     ,np.nanmean(my_specificity).flatten()
                     ,np.nanmean(num_components).flatten()
                     ,np.nanmean(in_inferred).flatten()
                     
                     ])





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
        f[target_str][:] = target.detach().cpu().numpy()
        f[predicted_segmentation_onehot_str][:]= curr.detach().cpu().numpy()
        f[data_str][:] = data.detach().cpu().numpy()


def save_for_metrics(epoch,target,output,data,log_every_n,batch_id,f,group_name):
    if(isinstance(output, list)):
        # list(map(lambda i: save_to_hdf5(f,i,group_name,batch_id,target[i],output[i],data[i]),range(len(output))))
        save_to_hdf5(f,0,group_name,batch_id,target[0],output[0],data)
    else:
        save_to_hdf5(f,0,group_name,batch_id,target,output,data)  


