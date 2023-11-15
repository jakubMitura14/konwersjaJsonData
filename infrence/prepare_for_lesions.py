import comet_ml
import multiprocessing
import optuna
import shutil
from time import sleep
from typing import Union, Tuple

import nnunetv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder, get_filenames_of_train_images_and_targets
from tqdm import tqdm
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)
from nnunetv2.utilities.label_handling.label_handling import LabelManager
import monai
import SimpleITK as sitk
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass
import lapgm
import tempfile

from pathlib import Path
from .anatomy_model.nnUNetTrainer.Main_trainer_pl import *
import torchio as tio
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import SimpleITK as sitk
import mdai
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import multiprocessing as mp
import functools
from functools import partial
import mdai
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
from scipy import ndimage

# import elastixRegister as elastixRegister
# from elastixRegister import reg_a_to_b,reg_a_to_b_be_meta_data
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
from pathlib import Path    
from .prprocess import *
from .prprocess import get_bool_arr_from_path,case_preprocessing,test_time_augmentation,my_to_float,get_el_1,save_label

"""
preparing the data for training on the lesions the plan for this file
1) get grouped paths with mri's and labels
2) run infrence of prostate anatomy save mean and std - use biggest connected component as a mask for mean
3) as preprocessing is a first step before infrence save also preprocessed files in hdf5
4) save attributes of each patient like psa, age, DRE, patient number into the hdf5 group
5) save sum of labels for each lesion from gold standard for each modality
6) report in separate csv dataset statistics like what percentae of a lesion annotation is outside of the gold standard anatomy segmentation; also results of the anatomical segmentations
 so dice and hausdorff distance for each modality
7) in case there was just a single annotator that had annotated given lesion make its center set value to 2 
"""


def groupByMaster(rowws):
    grouped_by_master= groupby(lambda row : row[1]['masterolds'],rowws)
    # grouped_by_master=[(key,list(group)) for key, group in grouped_by_master]
    return list(dict(grouped_by_master).items())



def get_thresholded(mean_tta,hparam_dict):
    mean_tta=(mean_tta>hparam_dict["treshold"]).astype(np.uint8)
    mean_tta=list(map( lambda i: get_largest_connected_component(mean_tta[i,:,:,:]),list(range(mean_tta.shape[0]))))
    mean_tta_bool= np.stack(mean_tta).astype(bool)
    return mean_tta_bool

def get_clinical_data(df,pat_num):
    our_prost_rows= df.iterrows()
    our_prost_rows= list(map(lambda tupl:tupl[1],our_prost_rows))
    row= list(filter(lambda roww : roww['patient_id']== pat_num,our_prost_rows))
    if(len(row)==0):            
        row= [{'dre_result':-1.0,'patient_age':-1.0, 'psa_result' :-1.0 }]
    clinical = list(map(lambda row : np.array([my_to_float(row['dre_result']),my_to_float(row['patient_age']),my_to_float(row['psa_result'])]), row))
    clinical= np.stack(clinical)
    clinical=np.nan_to_num(clinical, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)
    clinical=torch.tensor(clinical)
    return clinical

def get_lesion_paths(sourceFrame,input_names):
    """
    get paths related to cancer leasions
    """
    cols=sourceFrame.columns
    noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
    lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))
    filtered_per_modality=list(map(lambda name: list(filter(lambda el: name in el,lesion_cols)),input_names))
    return filtered_per_modality


def get_paths(group_dict,input_names):
    """
    get all necessary paths for anatomy infrence and lesion training
    """
    anatomic_cols=['cz_noSeg','pz_noSeg','sv_l_noSeg','sv_r_noSeg','pg_noSeg']
    #mri's
    
    mris=list(map(lambda inner_dict: inner_dict[1]['series_MRI_path'], group_dict))
    input_paths=list(map(lambda name: list(filter(lambda el: name in el,mris)),input_names))
    input_paths=list(map(get_el_1,input_paths))
    #anatomic gold standard
    anatomic_cols_paths=list(map( lambda col_name: list(map(lambda inner_dict: inner_dict[1][col_name], group_dict)),anatomic_cols))
    anatomic_cols_paths=list(map(lambda inner_list :list(filter(lambda el: len(el)>4,inner_list )) ,anatomic_cols_paths))
    if(len(anatomic_cols_paths[-1])==0 or len(anatomic_cols_paths[0])==0 or len(anatomic_cols_paths[1])==0 ):
        return " "
    anatomic_cols_paths= list(map(get_el_1,anatomic_cols_paths))
    filtered_per_modality=get_lesion_paths(sourceFrame,input_names)

    return input_paths,anatomic_cols_paths,filtered_per_modality

def get_anatomy_metrics(anatomic_cols_paths,mean_tta_bool,path_of_example,target_paths):
    """
    get anatomy metrics for each anatomical area and return dictionary with results
    results will be saved for future reference
    """
    ref_image=sitk.ReadImage(path_of_example)
    # pz=pz+cz
    pz=np.logical_or(get_bool_arr_from_path(anatomic_cols_paths[0],ref_image),get_bool_arr_from_path(anatomic_cols_paths[1],ref_image))
    full_pros=get_bool_arr_from_path(anatomic_cols_paths[4],ref_image)
    # tz is rest of prostate not pz
    tz=np.logical_and(np.logical_not(pz),full_pros)
    #sv jointly
    sv=np.logical_or(get_bool_arr_from_path(anatomic_cols_paths[2],ref_image),get_bool_arr_from_path(anatomic_cols_paths[3],ref_image))
    pz=sitk.GetArrayFromImage(sitk.ReadImage(target_paths[0])).astype(bool)
   
    # pz,full_pros,pz,sv    
    pz_metr=dict(get_Metrics(mean_tta_bool[0,:,:,:],pz,"pz")[0])
    full_metr=dict(get_Metrics(mean_tta_bool[3,:,:,:],full_pros,"full_pros")[0])

    tz_metr=dict(get_Metrics(mean_tta_bool[1,:,:,:],tz,"tz")[0])
    sv_metr=dict(get_Metrics(mean_tta_bool[2,:,:,:],sv,"sv")[0])

    #return dictionary with anatomy metrics merged
    return {**pz_metr, **tz_metr,**sv_metr,**full_metr}

def get_path(group_dict,key):
    """
    get valid path from group of rows
    """
    # group_dict=group_dict[1]
    key=key[4:]# removing suffix like t2w_ ...

    pp=list(map(lambda el: el[1][key], group_dict) )
    ll=list(filter(lambda el: len(el)>4,pp))
    if(len(ll)>0):
        return ll[0]
    return " "

def erode_if_needed(sum_of_lesions):
    """
    in case we do not have 2 annotations for a lesion the 'twos' will be the centers
    """
    #we check weather we have more than 1 anotation per lesion
    maxx= np.max(sum_of_lesions.flatten())
    #if we have less than 2 annotations we will simulate center by erosion
    if(maxx<2):
        arra=ndimage.binary_erosion(sum_of_lesions>0,iterations=1)
        arrb=ndimage.binary_erosion(sum_of_lesions>0,iterations=1)
        if(np.sum(arrb.flatten())>0):
            sum_of_lesions[arra]=2
            return sum_of_lesions       
        if(np.sum(arra.flatten())>0):
            sum_of_lesions[arrb]=2
            return sum_of_lesions       
    return sum_of_lesions

def process_labels_singleModality_single_lesion(modality_lesion_name,list_lesion_paths,mri_path,group_dict):
    """
    given single lesion associated with single modality it analyzes it from diffrent annotators and returns sum of lesions
    """
    list_lesion_paths=list(map(lambda k:get_path(group_dict,k) ,list_lesion_paths))
    list_lesion_paths=list(filter(lambda el: len(el)>4,list_lesion_paths))
    if(len(list_lesion_paths)==0):
        return (modality_lesion_name," ")
    list_lesion_paths=list(map(lambda p: get_bool_arr_from_path( p,sitk.ReadImage(mri_path)),list_lesion_paths))
    list_lesion_paths=list(map(lambda el: el.astype(np.uint8),list_lesion_paths))
    sum_of_lesions=np.sum(np.stack(list_lesion_paths),axis=0)
    sum_of_lesions=erode_if_needed(sum_of_lesions)
    
    


    return (modality_lesion_name,sum_of_lesions)

def process_labels_singleModality(modality_name,labels,mri_path,group_dict):
    """
    we group paths of a labels associated with a single modality and then we process them
    """
    label_names=list(map(lambda pathh:  
                         list(filter( lambda el_out:'/' not in el_out ,
                            list(filter(lambda el: 'lesion' in el  ,pathh.split('_')))))[0]
                         
                         ,labels ))
    label_names= list(filter( lambda el: '/' not in el ,label_names))
    # we want to get sum of all labels of the same name - so lesion 1 sum lesion 2 sum ...
    zipped_names= list(zip(label_names,labels))
    label_names_unique= np.unique(label_names)


    grouped_by_lesion_num=list(map(lambda lab_n:(lab_n, list(filter(lambda el: el[0]==lab_n,zipped_names ))) ,label_names_unique ))
    grouped_by_lesion_num=list(map( lambda el:(el[0],list(map(lambda el_in:f"{modality_name}_{el_in[1]}" ,el[1])))  ,grouped_by_lesion_num ))
    res=list(map(lambda pair : process_labels_singleModality_single_lesion(pair[0],pair[1],mri_path,group_dict),grouped_by_lesion_num ))
    res= filter(lambda el: el[1]!=" ",res)
    return (modality_name,res)


def process_lesion_labels(input_names,filtered_per_modality,mri_paths,group_dict):
    """
    process lesions labels for all modalities single study
    """
    filtered_per_modality_with_names=list(zip(input_names,filtered_per_modality,mri_paths))
    return list(map(lambda tri :process_labels_singleModality(tri[0],tri[1],tri[2],group_dict),filtered_per_modality_with_names))

def test_time_augment(data,plans_file,dataset_json_file,clinical,properties,checkpoint_paths,hparam_dict):
    """
    run test time augmentation on the ensemble of anatomy models
    """
    #prepare data for test time augmentation
    data=einops.rearrange(data,'c z y x->1 c z y x')
    fold=0
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)
    is_swin_monai=False

    mode_tta, mean_tta, std_tta, vvc_tta=test_time_augmentation(data
                           ,clinical
                           ,is_swin_monai
                           ,plans
                        ,configuration
                        ,fold,dataset_json
                        ,checkpoint_paths
                        ,properties
                        ,hparam_dict)
    return mean_tta, std_tta

def save_anatomy_to_hdf5(hdf5_group,anat_name,channell,mean_tta, std_tta,mean_tta_bool,debug_folder_name,mri_path):
    """
    save anatomy to hdf5
    """   
    save_label(mean_tta,channell,f"mean_{anat_name}",mri_path,debug_folder_name)
    save_label(std_tta,channell,f"std_{anat_name}",mri_path,debug_folder_name)
    save_label(mean_tta_bool.astype(np.uint8),channell,f"tresh_{anat_name}",mri_path,debug_folder_name)

    # hdf5_group.create_dataset(f"{anat_name}/mean_tta", data=mean_tta)
    # hdf5_group.create_dataset(f"{anat_name}/std_tta", data=std_tta)
    # hdf5_group.create_dataset(f"{anat_name}/mean_tta_bool", data=mean_tta_bool)
    
def save_lesions_to_hdf5(hdf5_group,lesion_arrs,debug_folder_name,mri_path):
    """
    save lesions labels to hdf5
    """
    
    for modality_name,modality_arrs in lesion_arrs:
        for modality_lesion_name,sum_of_lesions in modality_arrs:
            save_label(einops.rearrange(sum_of_lesions.astype(np.uint8),"a b c-> 1 a b c"),0,modality_lesion_name,mri_path,debug_folder_name)
        # hdf5_group.create_dataset(f"{lesion_name}_{i}", data=lesion_arrs[i])


def save_case_to_hdf5(plans_file,dataset_json_file,configuration, groupp,hparam_dict,df,checkpoint_paths):

    """
    main function for anatomy infrence and preparing data for lesion training
    """
    pat_num,listRows=groupp
    # get clinical data
    clinical=get_clinical_data(df,pat_num)

    # get mri and anatomic labels
    input_names=["t2w","adc","hbv"]
    input_paths,anatomic_cols_paths,filtered_per_modality=get_paths(listRows,input_names)
   

    

    # get data prepared for anatomy infrence - cropping, resampling, bias correction
    temp_dir ="/workspaces/konwersjaJsonData/data/debug"#tempfile.mkdtemp()# "/workspaces/konwersjaJsonData/data/curr" 
    data,properties,input_images_paths,target_paths=case_preprocessing(plans_file,dataset_json_file,configuration, input_paths,temp_dir,anatomic_cols_paths)
    #return if data issues
    if data==" ":
        return " "
    

    # we get lesion arrays for each modality and each lesion we sum for each lesion from diffrent annotators if there is just a single annotator on lesion the center will be artificially set to two
    lesion_arrs=process_lesion_labels(input_names,filtered_per_modality,input_paths,listRows)

    mean_tta, std_tta =test_time_augment(data,plans_file,dataset_json_file,clinical,properties,checkpoint_paths,hparam_dict)
    #thresholding and getting single largest component
    mean_tta_bool=get_thresholded(mean_tta,hparam_dict)
    #set all values outside of the largest component to 0
    # mean_tta[np.logical_not(mean_tta_bool)]=0

    #get anatomy metrics
    path_of_example=input_images_paths[0]  #"/workspaces/konwersjaJsonData/data/curr/1_t2w.nii.gz"
    anatomy_metrics=get_anatomy_metrics(anatomic_cols_paths,mean_tta_bool,path_of_example,target_paths)
    
    
    hdf5_group=[]
    debug_folder_name="/workspaces/konwersjaJsonData/data/debug"
    anat_dat=[("pz",0),("tz",1),("sv",2),("full_pros",3)]
    list(map(lambda el: save_anatomy_to_hdf5(hdf5_group,el[0],el[1],mean_tta, std_tta,mean_tta_bool,debug_folder_name,input_paths[0]),anat_dat))
    save_lesions_to_hdf5(hdf5_group,lesion_arrs,debug_folder_name,input_paths[0])



    #clear temporary directory
    # shutil.rmtree(temp_dir, ignore_errors=True)

    return anatomy_metrics






test_ids_CSVDir='/workspaces/konwersjaJsonData/test_ids.csv'
resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'

test_ids=pd.read_csv(test_ids_CSVDir)['ids'].to_numpy().flatten()
sourceFrame = pd.read_csv(resCSVDir)

modalities_of_intrest=['t2w','adc','hbv']

#clinical data dataframe
df=pd.read_csv("/workspaces/konwersjaJsonData/CRF.csv")
df=df[['patient_id','dre_result','patient_age','psa_result']].replace("Dodatni (+)", "1.0")
df=df[['patient_id','dre_result','patient_age','psa_result']].replace("Ujemny (-)", "0.0")
df['dre_result']=pd.to_numeric(df['dre_result'])
df['dre_result']=np.nan_to_num(df['dre_result'].to_numpy(),-1)


hparam_dict={}

hparam_dict["sigma_low"]=7.077968784031422#trial.suggest_float("sigma_low", 0.0,10.0)#5
hparam_dict["sigma_diff"]=8.048997012706618#trial.suggest_float("sigma_diff", 0.0,10.0)#2
hparam_dict["magnitude_range_low"]=3.85995070774654#trial.suggest_float("magnitude_range_low", 0.0,200.0)#50
hparam_dict["magnitude_range_diff"]=172.39161544780146#trial.suggest_float("magnitude_range_diff", 0.0,400.0)#100


hparam_dict["prob_elastic"]=1.0#trial.suggest_float("prob_elastic", 0.0,1.0)#1.0
hparam_dict["num_examples"]=30#trial.suggest_int("num_examples", 8,16)
hparam_dict["treshold"]=0.3#trial.suggest_float("treshold", 0.0,0.5)
hparam_dict["swin_weight"]=0.9#trial.suggest_float("swin_weight", 0.0,1.0)
plans_file = '/workspaces/konwersjaJsonData/infrence/plans/anatomy_plans.json'
dataset_json_file = '/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/dataset.json'
configuration = '3d_lowres'
checkpoint_paths=[(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_0/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_0/epoch=275-step=5796.ckpt",1.0)
                    # ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_0/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_1/epoch=509-step=10710.ckpt",1.0)
                    # ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_0/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_2/epoch=467-step=9828.ckpt",1.0)
                    # ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_2/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_3/epoch=413-step=8694.ckpt",1.0)
                    # ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_2/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_3/epoch=413-step=8694.ckpt",1.0)
                    # ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_3/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_4/epoch=449-step=9450.ckpt",1.0 )                     
                    
                    # ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_0/epoch=359-step=7561.ckpt",hparam_dict["swin_weight"] )     
                    # ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_1/epoch=539-step=11341.ckpt",hparam_dict["swin_weight"] )     
                    # ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_2/epoch=395-step=8317.ckpt",hparam_dict["swin_weight"] )     
                    # ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_3/epoch=323-step=6805.ckpt",hparam_dict["swin_weight"] )     
                    # ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_4/epoch=341-step=7183.ckpt",hparam_dict["swin_weight"] )     

                                                    ]


# filter_ids=lambda row: str(row[1]['masterolds']).strip() not in test_ids
grouped=toolz.pipe(sourceFrame.iterrows()
                                        ,filter(lambda row: row[1]['series_desc'] in modalities_of_intrest)
                                        # ,filter(filter_ids) # filter out all of the test cases
                                        ,groupByMaster)

# print(grouped)             
groupp=grouped[10]
save_case_to_hdf5(plans_file,dataset_json_file,configuration, groupp,hparam_dict,df,checkpoint_paths)                            

#python3 -m infrence.prepare_for_lesions