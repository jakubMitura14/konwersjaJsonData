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

sourceFrame = pd.read_csv(resCSVDir) 



cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))
main_modality = 'adc'




# modalities that we want to include in the model
modalities_of_intrest=['adc','hbv']

# prostate_col= 'pg_noSeg'
# new_col_name=prostate_col
# new_prost_col_name= 'inferred_pg'
# new_prost_parts__col_name= 'inferred_pg_parts'



non_mri_inputs=[]

# anatomic_cols=['afs_noSeg']



cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))

label_cols=lesion_cols
# label_cols=anatomic_cols+[prostate_col]
channel_names={  
    "0": "adc",
    "1": "hbv"  }





label_names= {  
    "background": 0,
    "lesion": 1,
    }

def process_labels_prim(labels,group,main_modality,label_new_path,zipped_modalit_path,out_pathsDict):
    labels= list(filter(lambda pathh : 'my_prost' not in  pathh, labels))
    label_names=list(map(lambda pathh: list(filter(lambda el: 'lesion' in el  ,pathh.split('_')[-5:]))[0] ,labels ))
    label_names= list(filter( lambda el: '/' not in el ,label_names))
    # we want to get sum of all labels of the same name - so lesion 1 sum lesion 2 sum ...
    # then we encode it in separate file 
    zipped_names= list(zip(label_names,labels))
    grouped_by_lesion_num = dict(groupby(lambda tupl :tupl[0],zipped_names)).items()
    grouped_by_lesion_num= list(map(lambda grouped :  (grouped[0],list(map(lambda el: el[1],grouped[1])) ) , grouped_by_lesion_num  ))
    grouped_by_lesion_num= list(map(lambda grouped :  (grouped[0],np.array(functools.reduce(get_bool_or, grouped[1]))),grouped_by_lesion_num  ))
    label_names_uniq=list(map(lambda el: el[0] ,grouped_by_lesion_num))
    label_nums= list(map(lambda name: list(filter(lambda el:el.isdigit(),name))  ,label_names_uniq))
    print(f"label_nums {label_nums} ")
    # we also need to increase number of copies of mri files so we will have one copy for each label
    label_new_paths= list(map(lambda lab_num : label_new_path.replace('0.nii.gz',f"{lab_num}.nii.gz")  ,label_nums))
    label_arrs= list(map(lambda tupl: tupl[1] ,grouped_by_lesion_num))
    
    zipped_new_labels= list(zip(label_new_paths,label_arrs  ))   
    #saving labels
    list(map(lambda tupl: save_from_arr(tupl[1],sitk.ReadImage(group[1][main_modality][0]),tupl[0]),zipped_new_labels))   
    # then for priming we do label erosion and then choose some random point 
    

    # we save result in zipped_modalit_path

    # for index,lab in enumerate(labels):        
    #     label_loc= label_new_path.replace('0.nii.gz',f"{index}.nii.gz")
    #     lab=get_bool_arr_from_path(lab)
    #     save_from_arr(lab,sitk.ReadImage(group[1][main_modality][0]),label_loc)
    
    # reduced = np.array(functools.reduce(get_bool_or, labels))
    # save_from_arr(reduced,sitk.ReadImage(group[1][main_modality][0]),label_new_path)
    
    zipped_modalit_path=list(map(zipped_modalit_path, label_new_paths))

    zipped_modalit_path= itertools.chain(*zipped_modalit_path)
    
    return [label_new_paths],zipped_modalit_path


def for_filter_unwanted(group):
    """ 
    we want only cases where  afs cz pz and tz are indicated
    """

    # print(f"tttt {group[1]['t2w'][1]}")
    # print(f"lll {len(group[1]['t2w'][1])}")

    # return len(group[1]['t2w'][1])==5
    return True


# grouped_rows= main_prepare_nnunet('287',modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame,main_modality,for_filter_unwanted)
grouped_rows= main_prepare_nnunet('287',modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame,main_modality,for_filter_unwanted,is_test_prep=False)





# mean dice on adc + hbv  0.7
# configurations=("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres")
#nnUNetv2_predict -i /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset287_Prostate/imagesTr -o /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/inferred_test_cases -d 287 -c '3d_fullres' 


# mainResults_folder="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate"
# CUDA_VISIBLE_DEVICES=0 nnUNet_results="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate" nnUNetv2_train 285 3d_cascade_fullres 0


# https://github.com/jakubMitura14/konwersjaJsonData.git

#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 287 3d_fullres 0


#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 287 3d_fullres 1
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 287 3d_fullres 2
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 287 3d_fullres 3
#CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 287 3d_fullres 4

# /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_preprocessed/Dataset281_Prostate/gt_segmentations

# cp -a /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/inferred_test_cases /workspaces/konwersjaJsonData/explore/to_eval_test_cases/inferred


# cp -a /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset287_Prostate /workspaces/konwersjaJsonData/explore/to_eval_test_cases/sources
# cp -a /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset287_Prostate /workspaces/konwersjaJsonData/explore/to_eval_test_cases/sources_detailed