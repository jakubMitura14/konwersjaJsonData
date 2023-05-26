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
dir_inferred_prost='/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_infered'
sourceFrame = pd.read_csv(resCSVDir) 

cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))
main_modality = 't2w'

# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']
prostate_col= 'pg_noSeg' # name of the column with segmentaton of whole prostate gland
#  'ob_noSeg' 'ob_Seg' 'ob_num' 'pg_noSeg'
#  'pg_Seg' 'pg_num' 'pz_noSeg' 'pz_Seg' 'pz_num' 'sv_l_noSeg' 'sv_l_Seg'
#  'sv_l_num' 'sv_r_noSeg' 'sv_r_Seg' 'sv_r_num' 'tz_noSeg' 'tz_Seg'
#  'tz_num' 'ur_noSeg' 'ur_Seg' 'ur_num'

non_mri_inputs=[prostate_col]

label_cols=lesion_cols
label_cols=lesion_cols+[prostate_col]
channel_names={  
    "0": "t2w", 
    "1": "adc",
    "2": "hbv",
    "3": "pg_noSeg"
    }
label_names= {  # THIS IS DIFFERENT NOW!
    "background": 0,
    "lesion": 1,
    }


def process_labels_prim(labels,group,main_modality,label_new_path):
    # we get the sum of all labels 
    # reduced = np.array(toolz.sandbox.parallel.fold(get_bool_or, labels,map=map))
    reduced = np.array(functools.reduce(get_bool_or, labels))
    # now we need to save the sumed label and all of the MRIs 
    # we want to make it compatible with both nnunet in general and with the picai dataset so we will keep picai convention of numering cases
    # 0 t2w, 1 adc 2 hbv additionally we will set prostate gland label as 3 which will be output of the segmentation algorithm passed as preprocessing step
    # in order to avoid problems with repeating ids all ids from 9
    # we need also to add related labels        
    save_from_arr(reduced,sitk.ReadImage(group[1][main_modality][0]),label_new_path)




grouped_rows= main_prepare_nnunet('279',modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame,main_modality)
    
# mainResults_folder="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate"
# CUDA_VISIBLE_DEVICES=0 nnUNet_results="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results/Dataset279_Prostate" nnUNetv2_train 279 3d_fullres 0
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 279 3d_fullres 0


# https://github.com/jakubMitura14/konwersjaJsonData.git