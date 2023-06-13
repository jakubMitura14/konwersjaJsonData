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

import elastixRegister as elastixRegister
from elastixRegister import reg_a_to_b

from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import os
from subprocess import Popen
import subprocess
from prepareNNunet import *

#metadata directory
resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
#directory with inferred prostates
dir_inferred_prost_parts='/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_parts_infered'
dir_inferred_prost='/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_infered'
new_prost_col_name= 'inferred_pg'
new_prost_parts__col_name= 'inferred_pg_parts'

sourceFrame = pd.read_csv(resCSVDir) 
new_col_names= ["new_afs","new_cz","new_pz","new_tz" ]



def get_id_from_file_name(path_str):
    path_str=path_str.replace('.nii.gz','')
    path_str=path_str[1:5]
    return int(path_str)

def add_t2w_to_name(source):
    if(source==' '):
        return ' '
    if('t2w' in source):
        return source
    new_path= source.replace('.nii.gz','_t2w.nii.gz')
    copy_changing_type(source, new_path)
    return new_path

def get_prost_part_separate_file(path,pros_part_name, curr_int):
    """ 
    given a path to file with multiple labels and int indicating ths label
    and a name we will create separate binary file for each label
    """
    if(path==" "):
        return " "
    writer = sitk.ImageFileWriter()
    newPathLab=path.replace('.nii.gz',f"_{pros_part_name}_t2w.nii.gz")
    image3D= sitk.ReadImage(path)
    arr=sitk.GetArrayFromImage(image3D)
    arr=(arr==curr_int)

    image = sitk.GetImageFromArray(arr.astype(np.uint8))  
 
    image.SetSpacing(image3D.GetSpacing())
    image.SetOrigin(image3D.GetOrigin())
    image.SetDirection(image3D.GetDirection())   
    image = sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    writer.SetFileName(newPathLab)
    writer.Execute(image)
    return newPathLab
    
def add_inferred_prost_to_dataframe(dir_inferred_prost_parts, df,new_col_name):
    """ 
    we have some inferred anatomical segmentations done by previous 
    models now we want to take the folder with 
    """
    list_files= os.listdir(dir_inferred_prost_parts)
    list_files= list(filter(lambda el : el[0]=='9' ,list_files ))
    list_ids= list(map(get_id_from_file_name,list_files))
    list_files= list(map( lambda el: f"{dir_inferred_prost_parts}/{el}" ,list_files))
    file_and_id= dict(list(zip(list_ids,list_files)))
    new_col_dat= list(map( lambda el: file_and_id.get(el,' ') ,df['masterolds'].to_numpy() ))
    #changing path name to mark it is t2w related
    new_col_dat= list(map(add_t2w_to_name,new_col_dat))

    df[new_col_name]=new_col_dat
    return df



def add_inferred_prost_parts_to_dataframe(dir_inferred_prost_parts, df,pros_part_name,curr_int):
    """ 
    we have some inferred anatomical segmentations done by previous 
    models now we want to take the folder with 
    """
    list_files= os.listdir(dir_inferred_prost_parts)
    list_files= list(filter(lambda el : el[0]=='9' ,list_files ))
    list_ids= list(map(get_id_from_file_name,list_files))
    list_files= list(map( lambda el: f"{dir_inferred_prost_parts}/{el}" ,list_files))
    file_and_id= dict(list(zip(list_ids,list_files)))
    new_col_dat= list(map( lambda el: file_and_id.get(el,' ') ,df['masterolds'].to_numpy() ))
    
    #changing path name to mark it is t2w related
    new_col_dat= list(map(lambda path:get_prost_part_separate_file(path,pros_part_name, curr_int),new_col_dat))

    df[pros_part_name]=new_col_dat
    return df




cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))
main_modality = 't2w'


#adding data about earlier inferred anatomy
sourceFrame=add_inferred_prost_to_dataframe(dir_inferred_prost, sourceFrame,new_prost_col_name)
sourceFrame=add_inferred_prost_parts_to_dataframe(dir_inferred_prost_parts, sourceFrame,new_col_names[0], 1)
sourceFrame=add_inferred_prost_parts_to_dataframe(dir_inferred_prost_parts, sourceFrame,new_col_names[1], 2)
sourceFrame=add_inferred_prost_parts_to_dataframe(dir_inferred_prost_parts, sourceFrame,new_col_names[2], 3)
sourceFrame=add_inferred_prost_parts_to_dataframe(dir_inferred_prost_parts, sourceFrame,new_col_names[3], 4)

sourceFrame=sourceFrame.loc[sourceFrame[new_prost_col_name] != " "]
sourceFrame=sourceFrame.loc[sourceFrame[new_col_names[0]] != " "]
sourceFrame=sourceFrame.loc[sourceFrame[new_col_names[1]] != " "]
sourceFrame=sourceFrame.loc[sourceFrame[new_col_names[2]] != " "]
sourceFrame=sourceFrame.loc[sourceFrame[new_col_names[3]] != " "]




# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']

# prostate_col= 'pg_noSeg'
# new_col_name=prostate_col
# new_prost_col_name= 'inferred_pg'
# new_prost_parts__col_name= 'inferred_pg_parts'



non_mri_inputs=[new_prost_col_name]+new_col_names

# anatomic_cols=['afs_noSeg']



cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))

label_cols=lesion_cols
# label_cols=anatomic_cols+[prostate_col]
channel_names={  
    "0": "t2w", 
    "1": "adc",
    "2": "hbv",
    "3": new_prost_col_name,
    "4": new_col_names[0],
    "5": new_col_names[1],
    "6": new_col_names[2],
    "7": new_col_names[3]    }





label_names= {  
    "background": 0,
    "lesion": 1,
    }

def process_labels_prim(labels,group,main_modality,label_new_path):
    reduced = np.array(functools.reduce(get_bool_or, labels))
    # now we need to save the sumed label and all of the MRIs 
    # we want to make it compatible with both nnunet in general and with the picai dataset so we will keep picai convention of numering cases
    # 0 t2w, 1 adc 2 hbv additionally we will set prostate gland label as 3 which will be output of the segmentation algorithm passed as preprocessing step
    # in order to avoid problems with repeating ids all ids from 9
    # we need also to add related labels        
    save_from_arr(reduced,sitk.ReadImage(group[1][main_modality][0]),label_new_path)


def for_filter_unwanted(group):
    """ 
    we want only cases where  afs cz pz and tz are indicated
    """

    # print(f"tttt {group[1]['t2w'][1]}")
    # print(f"lll {len(group[1]['t2w'][1])}")

    # return len(group[1]['t2w'][1])==5
    return True


grouped_rows= main_prepare_nnunet('283',modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame,main_modality,for_filter_unwanted)

#nnUNetv2_predict -i /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset281_Prostate/imagesTr -o /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_parts_infered -d 281 -c '3d_fullres' 


# mainResults_folder="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate"
# CUDA_VISIBLE_DEVICES=0 nnUNet_results="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate" nnUNetv2_train 279 3d_fullres 0
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 281 3d_fullres 0


# https://github.com/jakubMitura14/konwersjaJsonData.git

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 283 3d_fullres 0


#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 283 3d_fullres 1
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 283 3d_fullres 2
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 283 3d_fullres 3
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 283 3d_fullres 4

# /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_preprocessed/Dataset281_Prostate/gt_segmentations
