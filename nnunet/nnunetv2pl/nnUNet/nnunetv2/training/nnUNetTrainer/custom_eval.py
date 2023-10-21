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
import math
from sklearn.metrics import auc, precision_recall_curve, roc_curve

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

def get_dice_single(one_hot,target):
    labelPred=sitk.GetImageFromArray(one_hot.astype(np.uint8))
    labelTrue=sitk.GetImageFromArray(target.astype(np.uint8))
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    return dicecomputer.GetDiceCoefficient()

def get_dice_lesions(arrs):

    curr_in,centers,inferred,big_mask,data=arrs
    

    return [get_dice_single(inferred,centers),get_dice_single(inferred,big_mask)]




def is_sth_in_areas(uniq_num,arr,inferred,curr_twos):

    bool_arr=(arr.copy()==uniq_num)
    summ=np.sum(inferred[bool_arr].flatten())
    # res= summ> (np.sum(bool_arr.flatten())//2)
    
    curr_twos_loc=curr_twos.astype(bool)
    curr_twos_loc= np.logical_and(curr_twos_loc,bool_arr)

    # summ=np.sum(inferred[curr_twos_loc].flatten())
    # res_b = summ> (np.sum(curr_twos_loc.flatten())//2)
    res= summ> (np.sum(curr_twos_loc.flatten())//2)
    return res

def get_connected_components_num(arr):
    connected=sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(arr.astype(int))))
    uniqq=np.unique(connected)
    uniqq= list(filter(lambda el:el>0,uniqq))
    return len(uniqq)    

def get_my_sensitivity(arrs):
    curr_in,curr_twos,inferred,curr_bigger_mask,data=arrs
    
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


    res=np.array(list(map(lambda uniq_num: is_sth_in_areas(uniq_num,connected,inferred,curr_twos),uniqq)))
    res=res.astype(int)
    return np.mean(res.flatten())

def save_single_arr(image_array,batch_idd, bn, c,for_explore,name,typee ,metr):
    # image_array,batch_idd, bn, c,for_explore,name,typee=args
    # im= sitk.GetImageFromArray(image_array)
    folder=f"{for_explore}/{metr}_{(int(batch_idd)*100)+int(bn)}" #/{name}_{c}
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
    dice=get_dice_lesions(arrs)
    # print(f"ddddd dice {dice}")
    # print(f"specificity {specificity} sensitivity {sensitivity}")
    curr_in,curr_twos,inferred,curr_bigger_mask,data =arrs
    #if(False):
    if(to_save_files):
        # print(f"fffffffffffff curr_bigger_mask {np.sum(curr_bigger_mask.flatten())} curr_twos {np.sum(curr_twos.flatten())} ")
        save_single_arr(curr_in,batch_idd, bn, 0,for_explore,"curr_in",np.uint8 ,sensitivity)
        save_single_arr(curr_twos,batch_idd, bn, 0,for_explore,"curr_twos",np.uint8 ,sensitivity)
        save_single_arr(inferred,batch_idd, bn, 0,for_explore,"inferred",np.uint8 ,sensitivity)
        save_single_arr(curr_bigger_mask,batch_idd, bn, 0,for_explore,"curr_bigger_mask",np.uint8 ,sensitivity)

        save_single_arr(data[0,:,:,:],batch_idd, bn, 0,for_explore,"adc",float,sensitivity)
        save_single_arr(data[1,:,:,:],batch_idd, bn, 1,for_explore,"hbv",float,sensitivity)
        save_single_arr(data[2,:,:,:],batch_idd, bn, 2,for_explore,"t2w",float,sensitivity)
        save_single_arr(data[3,:,:,:],batch_idd, bn, 3,for_explore,"data",float,sensitivity)
        save_single_arr(data[4,:,:,:],batch_idd, bn, 4,for_explore,"data",float,sensitivity)
        if(data.shape[0]==6):
            save_single_arr(data[5,:,:,:],batch_idd, bn, 5,for_explore,"priming",float,sensitivity)

    return specificity,sensitivity,num_components,np.sum(inferred.flatten()),dice[0],dice[1]


def concat_local(batch_ids,f,group_name,name):
    res=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/{name}"][:,:,:,:],batch_ids))
    return np.concatenate(res,axis=0)
 
def concat_local_data(batch_ids,f,group_name,name):
    res=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/{name}"][:,:,:,:,:],batch_ids))
    return np.concatenate(res,axis=0)
    

def calc_custom_metrics(group_name,f,for_explore,to_save_files,anatomy_metr=False, batch_size=1):    
    try: 
        batch_nums= np.array(list(f[group_name].keys()))
        # print(f"111 batch_nums {batch_nums} group_name {group_name}")
    
        if(batch_nums.shape[0]<20):
            chunk_size = 1
            batch_nums=np.array_split(batch_nums, math.ceil(batch_nums.shape[0]/chunk_size))
        else:
            chunk_size=max(10//batch_size,1)
            batch_nums=np.array_split(batch_nums, math.ceil(batch_nums.shape[0]/chunk_size))
        
        # target=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/target"][:,:,:,:], batch_nums))
        # predicted_segmentation_onehot=list(map(lambda batch_id :f[f"{group_name}/{batch_id}/predicted_segmentation_onehot"][:,:,:,:], batch_nums))

        # target=np.concatenate(target,axis=0)
        # predicted_segmentation_onehot=np.concatenate(predicted_segmentation_onehot,axis=0)
        if(to_save_files):
            os.makedirs(for_explore,exist_ok=True)
            shutil.rmtree(for_explore,ignore_errors=True)   
            os.makedirs(for_explore,exist_ok=True)

        tempdir='/workspaces/konwersjaJsonData/explore/temp_csv'
        shutil.rmtree(tempdir,ignore_errors=True)       
        os.makedirs(tempdir,exist_ok=True)
        

        if(anatomy_metr):
            res= list(map(lambda batch_ids: calc_custom_metrics_inner(concat_local_data(batch_ids,f,group_name,"target")
                                                                    ,concat_local_data(batch_ids,f,group_name,"predicted_segmentation_onehot")
                                                                    ,concat_local_data(batch_ids,f,group_name,"data")
                                                                    ,f,for_explore,to_save_files,batch_ids,anatomy_metr=anatomy_metr,tempdir=tempdir,group_name=group_name),batch_nums))
        else:
            res= list(map(lambda batch_ids: calc_custom_metrics_inner(concat_local_data(batch_ids,f,group_name,"target")
                                                                    ,concat_local(batch_ids,f,group_name,"predicted_segmentation_onehot")
                                                                    ,concat_local_data(batch_ids,f,group_name,"data")
                                                                    ,f,for_explore,to_save_files,batch_ids,anatomy_metr=anatomy_metr,tempdir=tempdir,group_name=group_name),batch_nums))
        if(anatomy_metr):
            res= list(itertools.chain(*res))
            res= list(itertools.chain(*res))

            grouped_by_metr_name=  list(dict(groupby(lambda row : row[0],res)).items())
            grouped_by_metr_name =list(map(lambda tupl: (tupl[0],list(map(lambda inner_tupl: inner_tupl[1], tupl[1])) )   ,grouped_by_metr_name))
            grouped_by_metr_name =list(map(lambda tupl: (tupl[0],np.nanmean(np.array(tupl[1])))  ,grouped_by_metr_name))



            # filtered=list(map(lambda name: list(filter(lambda tupl: tupl[0]==name ,res ))  , metrics_names))
            # filtered= list(map( lambda listt: np.nanmean(np.array(list(map(lambda tupl :tupl[1] ,listt )))) ,filtered))
            return grouped_by_metr_name
        res=np.concatenate(res,axis=-1)
        # res= list(map(lambda batch_ids: calc_custom_metrics_inner(f[f"{group_name}/{batch_id}/target"][:,:,:,:],f[f"{group_name}/{batch_id}/predicted_segmentation_onehot"][:,:,:,:]),batch_nums))
        res= np.nanmean(res,axis=-1)
        res= np.nan_to_num(res,posinf=0.0, neginf=0.0)
        return res
    except:
        return np.zeros(8)-0.1

def prep_arr_list(inn,twos,curr,bigger_mask,data,batch_num):
    
    return list(map(lambda bi: (inn[bi,:,:,:],twos[bi,:,:,:],curr[bi,:,:,:],bigger_mask[bi,:,:,:],data[bi,:,:,:,:]  ) ,range(batch_num)))


def prep_arr_list_anatomy(predicted_segmentation_onehot,target,data,batch_num,batch_idd):
    return list(map(lambda bi: ((batch_idd*100)+bi,predicted_segmentation_onehot[bi,:,:,:,:],target[bi,:,:,:],data[bi,:,:,:,:]) ,range(data.shape[0])))

def save_arrs_anatomy(predicted_segmentation_onehot,data,target,batch_idd,for_explore,hd):
    save_single_arr(predicted_segmentation_onehot[0,:,:,:],batch_idd, 0, 0,for_explore,"inferred_pz",np.uint8 ,hd)
    save_single_arr(predicted_segmentation_onehot[1,:,:,:],batch_idd, 0, 0,for_explore,"inferred_tz",np.uint8 ,hd)
    save_single_arr(predicted_segmentation_onehot[2,:,:,:],batch_idd, 0, 0,for_explore,"inferred_sv",np.uint8 ,hd)
    save_single_arr(predicted_segmentation_onehot[3,:,:,:],batch_idd, 0, 0,for_explore,"inferred_sum",np.uint8 ,hd)

    save_single_arr(target[0,:,:,:],batch_idd, 0, 0,for_explore,"target_pz",np.uint8,hd )
    save_single_arr(target[1,:,:,:],batch_idd, 0, 0,for_explore,"target_tz",np.uint8,hd )
    save_single_arr(target[2,:,:,:],batch_idd, 0, 0,for_explore,"target_sv",np.uint8,hd )
    save_single_arr(target[3,:,:,:],batch_idd, 0, 0,for_explore,"target_sum",np.uint8,hd )
    save_single_arr(data[0,:,:,:],batch_idd, 0, 0,for_explore,"t2w",float,hd )

# def prep_anatomy_target(target):
#     return target[1,:,:,:].astype(np.uint8)+target[2,:,:,:].astype(np.uint8)*2

def get_largest_connected_component(binary_image):
    if(np.sum(binary_image.flatten())==0):
        return binary_image
    binary_image=sitk.GetImageFromArray(binary_image.astype(np.uint8))
    #taken from https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958
    # 1. Convert binary image into a connected component image, each component has an integer label.
    # 2. Relabel components so that they are sorted according to size (there is an
    #    optional minimumObjectSize parameter to get rid of small components).
    # 3. Get largest connected componet, label==1 in sorted component image.
    component_image = sitk.ConnectedComponent(binary_image)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    return sitk.GetArrayFromImage(largest_component_binary_image)
    


def get_Metrics(one_hot,target,name):
    labelPred=sitk.GetImageFromArray(one_hot.astype(np.uint8))
    labelTrue=sitk.GetImageFromArray(target.astype(np.uint8))
    quality=dict()
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality[f"dice_{name}"]=dicecomputer.GetDiceCoefficient()
    quality[f"volume_similarity_{name}"]=dicecomputer.GetVolumeSimilarity()

    if(np.sum(one_hot.flatten())==0 or np.sum(target.flatten())==0):
        return list(quality.items()),quality[f"dice_{name}"]

    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality[f"avgHausdorff_{name}"]=hausdorffcomputer.GetAverageHausdorffDistance()
    # quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    return list(quality.items()),quality[f"dice_{name}"]

def evaluate_single_anatomy_case(arrs,tempdir,for_explore,to_save_files):
    bi,predicted_segmentation_onehot,target,data=arrs
    
    predicted_segmentation_onehot=list(map(lambda i: get_largest_connected_component(predicted_segmentation_onehot[i,:,:,:]) , range(predicted_segmentation_onehot.shape[0])))
    predicted_segmentation_onehot= np.stack(predicted_segmentation_onehot)
    
    pz_metr,hd = get_Metrics(predicted_segmentation_onehot[0,:,:,:],target[0,:,:,:],'pz')
    tz_metr,hd = get_Metrics(predicted_segmentation_onehot[1,:,:,:],target[1,:,:,:],'tz')
    sv_metr,hd = get_Metrics(predicted_segmentation_onehot[2,:,:,:],target[2,:,:,:],'sv')
    mean_metr,hd = get_Metrics(predicted_segmentation_onehot[3,:,:,:],target[3,:,:,:],'all')
    
    res= list(itertools.chain(*[pz_metr,tz_metr,mean_metr,sv_metr]))
    if(to_save_files):
        save_arrs_anatomy(predicted_segmentation_onehot,data,target,bi,for_explore,hd)  
    
    return res



def calc_custom_metrics_inner(target,predicted_segmentation_onehot,data,f,for_explore,to_save_files,batch_ids,anatomy_metr,tempdir,group_name):
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

        anatomy_arrs=prep_arr_list_anatomy(predicted_segmentation_onehot,target,data,shapp[0],batch_idd)
        res=list(map(partial(evaluate_single_anatomy_case,tempdir=tempdir,to_save_files=to_save_files,for_explore=for_explore),anatomy_arrs))
        return res
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
    dice_centers=ress[:,4]
    dice_all=ress[:,5]
    

    my_sensitivity=list(filter(lambda el: np.array(el).flatten()[0]>-1,my_sensitivity  ))      
    if(len(my_sensitivity)>0):
        my_sensitivity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_sensitivity))))
        my_specificity= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),my_specificity))))
        num_components= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),num_components))))
        dice_centers= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),dice_centers))))
        dice_all= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),dice_all))))
        in_inferred= np.nanmean(np.array(list(map(lambda el : np.array(np.nanmean(el)).flatten(),in_inferred))))
        is_correct= (my_sensitivity*2+my_specificity)/3
    
  
    is_correct=np.array(is_correct).flatten()

    my_sensitivity=np.array(np.nanmean(np.array(my_sensitivity).flatten()))
    my_specificity=np.array(np.nanmean(np.array(my_specificity).flatten()))
    
    num_components=np.array(np.nanmean(np.array(num_components).flatten()))
    in_inferred=np.array(np.nanmean(np.array(in_inferred).flatten()))
    dice_centers=np.array(np.nanmean(np.array(dice_centers).flatten()))
    dice_all=np.array(np.nanmean(np.array(dice_all).flatten()))


    return np.array([np.nanmean(percent_in).flatten()
                     ,np.nanmean(percent_out).flatten()
                     ,np.nanmean(percent_covered).flatten()
                     ,np.nanmean(is_correct).flatten()
                     ,np.nanmean(my_sensitivity).flatten()
                     ,np.nanmean(my_specificity).flatten()
                     ,np.nanmean(num_components).flatten()
                     ,np.nanmean(in_inferred).flatten()
                     ,np.nanmean(dice_centers).flatten()                     
                     ,np.nanmean(dice_all).flatten()                     
                     ])



def get_pred_one_hot(output,is_regions):
    if(is_regions):
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        return predicted_segmentation_onehot
    else:
        
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        return predicted_segmentation_onehot
        

def save_to_hdf5(f,inner_id,group_name,batch_id,target,output,data):
    output_0_w=float(os.getenv('output_0_w'))
    output_1_w=float(os.getenv('output_1_w'))
    sum_w=output_0_w+output_1_w

    
    output=torch.stack([output[:,0,:,:,:],(output[:,2,:,:,:]*output_1_w+output[:,1,:,:,:]*output_0_w) ],dim=1)
    predicted_segmentation_onehot=get_pred_one_hot(output,False)
    
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


def save_to_hdf5_anatomy(f,inner_id,group_name,batch_id,target,output,data):
    
    predicted_segmentation_onehot=get_pred_one_hot(output,True)
    curr=predicted_segmentation_onehot.round().bool()
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

def get_first_if_list(inn):
    if(isinstance(inn, list)):
        return inn[0]
    return inn

def save_for_metrics(epoch,target,output,data,log_every_n,batch_id,f,group_name,is_anatomy_metr=False):
    # list(map(lambda i: save_to_hdf5(f,i,group_name,batch_id,target[i],output[i],data[i]),range(len(output))))
    if(is_anatomy_metr):
        save_to_hdf5_anatomy(f,0,group_name,batch_id,get_first_if_list(target),get_first_if_list(output),data)
    else:    
        save_to_hdf5(f,0,group_name,batch_id,get_first_if_list(target),get_first_if_list(output),data)


