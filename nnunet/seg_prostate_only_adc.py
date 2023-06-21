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
main_modality = 't2w'


# modalities that we want to include in the model
modalities_of_intrest=['adc']

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
    
 }





label_names= {  
    "background": 0,
    "lesion": 1,
    }

def process_labels_prim(labels,group,main_modality,label_new_path):
    labels= list(filter(lambda pathh : 'my_prost' not in  pathh, labels))
    labels= list(filter(lambda pathh : 'adc' in  pathh, labels))
    
    print(labels)
    
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


grouped_rows= main_prepare_nnunet('284',modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame,main_modality,for_filter_unwanted)


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


# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0000.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0000.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0001.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0001.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0002.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0002.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0003.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0003.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0004.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0004.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0005.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0005.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0006.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0006.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9000100_0007.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0007.nii.gz

# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/labelsTr/9000100.nii.gz /workspaces/konwersjaJsonData/explore/imagess/label_9000100.nii.gz



# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0000.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9000100_0000.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0001.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9015900_0001.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0002.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9015900_0002.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0003.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9015900_0003.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0004.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9015900_0004.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0005.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9015900_0005.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0006.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9015900_0006.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9015900_0007.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9015900_0007.nii.gz

# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/labelsTr/9015900.nii.gz /workspaces/konwersjaJsonData/explore/imagess/label_9015900.nii.gz


# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0000.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0000.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0001.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0001.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0002.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0002.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0003.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0003.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0004.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0004.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0005.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0005.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0006.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0006.nii.gz
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/imagesTr/9005500_0007.nii.gz /workspaces/konwersjaJsonData/explore/imagess/9005500_0007.nii.gz

# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset283_Prostate/labelsTr/9005500.nii.gz /workspaces/konwersjaJsonData/explore/imagess/label_9005500.nii.gz

