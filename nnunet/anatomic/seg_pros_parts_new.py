"""
segmenation where target is the sum of t2w,adc,hbv labels and we add as input additionally whole prostate segmentations

"""

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


from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import os
from subprocess import Popen
import subprocess
from prepare_nnunet_anatomic import *

#metadata directory
resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
#directory with inferred prostates

# dir_inferred_prost='/home/sliceruser/workspaces/konwersjaJsonData/my_prost_infered'

# dir_inferred_prost='/workspaces/prost_anatomy_seg/data/my_prost_infered'
sourceFrame = pd.read_csv(resCSVDir) 
# new_col_name= 'inferred_pg'



def get_id_from_file_name(path_str):
    path_str=path_str.replace('.nii.gz','')
    path_str=path_str[1:5]
    return int(path_str)

# def add_t2w_to_name(source):
#     if(source==' '):
#         return ' '
#     # if('t2w' in source):
#     #     return source
#     # new_path= source.replace('.nii.gz','_t2w.nii.gz')
#     # copy_changing_type(source, new_path)
#     # return new_path
#     return source
    

# def add_inferred_full_prost_to_dataframe(dir_inferred_prost, df,new_col_name):
#     """ 
#     we have some inferred anatomical segmentations done by previous 
#     models now we want to take the folder with 
#     """
#     list_files= os.listdir(dir_inferred_prost)
#     list_files= list(filter(lambda el : el[0]=='9' ,list_files ))
#     list_ids= list(map(get_id_from_file_name,list_files))
#     list_files= list(map( lambda el: f"{dir_inferred_prost}/{el}" ,list_files))
#     file_and_id= dict(list(zip(list_ids,list_files)))
#     new_col_dat= list(map( lambda el: file_and_id.get(el,' ') ,df['masterolds'].to_numpy() ))
#     #changing path name to mark it is t2w related
#     new_col_dat= list(map(add_t2w_to_name,new_col_dat))

#     df[new_col_name]=new_col_dat

#     return df


cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))
main_modality = 't2w'

# sourceFrame=add_inferred_full_prost_to_dataframe(dir_inferred_prost, sourceFrame,new_col_name)
# modalities that we want to include in the model
main_modality = 't2w'
modalities_of_intrest=['t2w','adc','hbv']

# prostate_col= 'pg_noSeg'
# new_col_name=prostate_col

# new_col_name= 'inferred_pg'

non_mri_inputs=[]
# prostate_col= new_col_name # name of the column with segmentaton of whole prostate gland

anatomic_cols=['afs_noSeg','cz_noSeg','pz_noSeg','tz_noSeg']
# anatomic_cols=['afs_noSeg']

label_cols=anatomic_cols
# label_cols=anatomic_cols+[prostate_col]
channel_names={  
    "0": "t2w", 
    "1": "adc", 
    "2": "hbv", 
    }


label_names= {  
    "background": 0,
    "pz": 1,
    "tz": 2,
    "full_prost":[1,2]
    }

# label_names= {  # THIS IS DIFFERENT NOW!
#     "background": 0,
#     "afs": 1,
#     }

def get_int_arr_from_path(pathh):
    """
    given path reads it and return associated array
    then it casts it to boolean data type
    """
    index=15
    to_ignore=False
    if('afs' in pathh):
        index=2
    elif('cz' in pathh):
        index=1        
    elif('pz' in pathh):
        index=1
    elif('tz' in pathh):
        index=2
    else:
        to_ignore=True

    imageA=sitk.ReadImage(pathh)
    imageA=sitk.GetArrayFromImage(imageA)
    if(to_ignore):
        return np.zeros_like(imageA)
    return np.array(imageA.astype(bool).astype(np.uint8) *(index))

def process_labels_prim(labels,group,main_modality,label_new_path,zipped_modalit_path,out_pathsDict):
    # we get the sum of all labels 
    arrays= list(map(get_int_arr_from_path,labels))
    # arrays= list(map(list,arrays))
    reduced = np.sum(np.stack(arrays,axis=0),axis=0).astype(np.uint8)
    print(np.unique(reduced))
    save_from_arr(reduced,sitk.ReadImage(group[1][main_modality][0]),label_new_path)
    return [label_new_path],zipped_modalit_path


def for_filter_unwanted(group):
    """ 
    we want only cases where  afs cz pz and tz are indicated
    """

    # print(f"tttt {group[1]['t2w'][1]}")
    # print(f"lll {len(group[1]['t2w'][1])}")

    # return len(group[1]['t2w'][1])==5
    return True


grouped_rows= main_prepare_nnunet('294',modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame,main_modality,for_filter_unwanted)

#my_proj_name="seg debug" tag="pc debug" my_proj_desc="pc debug" nnUNetv2_train 294 3d_lowres 0 -tr My_Anatomy_trainer

#nnUNetv2_predict -i /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset281_Prostate/imagesTr -o /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_parts_infered -d 281 -c '3d_fullres' 


# mainResults_folder="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate"
# CUDA_VISIBLE_DEVICES=0 nnUNet_results="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate" nnUNetv2_train 279 3d_fullres 0
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 294 3d_fullres 0


# https://github.com/jakubMitura14/konwersjaJsonData.git

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 294 3d_fullres 1
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 294 3d_fullres 2
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 294 3d_fullres 3
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 294 3d_fullres 4

# /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_preprocessed/Dataset281_Prostate/gt_segmentations
