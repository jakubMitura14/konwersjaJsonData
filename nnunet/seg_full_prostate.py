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
from datetime import date
from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import os
from subprocess import Popen
import subprocess
from nnunet.prepareNNunet import *

resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'

sourceFrame = pd.read_csv(resCSVDir) 
cols=sourceFrame.columns
main_modality = 't2w'

# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']
prostate_col= 'pg_noSeg' # name of the column with segmentaton of whole prostate gland
#  'ob_noSeg' 'ob_Seg' 'ob_num' 'pg_noSeg'
#  'pg_Seg' 'pg_num' 'pz_noSeg' 'pz_Seg' 'pz_num' 'sv_l_noSeg' 'sv_l_Seg'
#  'sv_l_num' 'sv_r_noSeg' 'sv_r_Seg' 'sv_r_num' 'tz_noSeg' 'tz_Seg'
#  'tz_num' 'ur_noSeg' 'ur_Seg' 'ur_num'

non_mri_inputs=[]

label_cols=[prostate_col]
channel_names={  
    "0": "t2w", 
    "1": "adc",
    "2": "hbv",
    }
label_names= {  # THIS IS DIFFERENT NOW!
    "background": 0,
    "prostate": 1,
    }
def process_labels_prim(labels,group,main_modality,label_new_path):
    copy_changing_type(labels[0], label_new_path)



dataset_id='280'

# grouped_rows= main_prepare_nnunet(dataset_id,modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame,main_modality)
    

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 280 3d_fullres 0

def train_2_folds_and_save():
    """
    runs the training on 2 folds and copies resulting checkpoints into a separate folder
    """
    main_dir='/home/sliceruser/workspaces/konwersjaJsonData/results_to_save'
    os.makedirs(main_dir ,exist_ok = True)
    cmd_terminal=f"CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {dataset_id} 3d_fullres 2"
    p = Popen(cmd_terminal, shell=True)
    p.wait()
    cmd_terminal=f"CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {dataset_id} 3d_fullres 3"
    p = Popen(cmd_terminal, shell=True)
    p.wait()
    cmd_terminal=f"CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {dataset_id} 3d_fullres 4"
    p = Popen(cmd_terminal, shell=True)
    p.wait()

    dir_project_id=f"/home/sliceruser/workspaces/konwersjaJsonData/results_to_save/{dataset_id}"
    os.makedirs(dir_project_id ,exist_ok = True)
    dir_project_id_with_date=f"/home/sliceruser/workspaces/konwersjaJsonData/results_to_save/{dataset_id}/{date.today()}"
    # os.makedirs(dir_project_id_with_date ,exist_ok = True)
    original_results_folder="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results"
    shutil.copytree(original_results_folder, dir_project_id_with_date,dirs_exist_ok=True) 

train_2_folds_and_save()

nnUNetv2_find_best_configuration 280 -c CONFIGURATIONS 
nnUNetv2_find_best_configuration 280 -c '3d_fullres' 

# nnUNetv2_find_best_configuration 280 -c '3d_fullres' -f 0 -np 8
# nnUNetv2_find_best_configuration -h
#The file "/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset280_Prostate/labelsTr/9043100.nii.gz" does not exist.

nnUNetv2_predict -i /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset280_Prostate/imagesTr -o /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_infered -d 280 -c '3d_fullres' 

--save_probabilities
9000100_0000.nii.gz  9014100_0001.nii.gz

cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset280_Prostate/imagesTr/9000100_0003.nii.gz /workspaces/konwersjaJsonData/explore/in_prost_seg/9000100_0003.nii.gz