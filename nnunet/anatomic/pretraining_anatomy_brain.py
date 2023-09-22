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

# from elastixRegister import reg_a_to_b,reg_a_to_b_be_meta_data,reg_a_to_b_by_metadata_single_b
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
import os
import subprocess
from pathlib import Path

import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)

dataset_id=294
nNunetBaseFolder='/home/sliceruser/nnunetMainFolder'
os.makedirs(f"{nNunetBaseFolder}/nnUNet_preprocessed" ,exist_ok = True)
os.makedirs(f"{nNunetBaseFolder}/nnUNet_raw" ,exist_ok = True)
shutil.rmtree(f"{nNunetBaseFolder}/nnUNet_preprocessed")
shutil.rmtree(f"{nNunetBaseFolder}/nnUNet_raw")


taskName= f"Dataset{dataset_id}_Prostate"
taskFolder = join("/home/sliceruser/preprocessed/anatomic_preprocessed",taskName)
preprocesss_folder= join(nNunetBaseFolder,'nnUNet_preprocessed')
results_folder= join(nNunetBaseFolder,'nnUNet_results')
mainResults_folder="/home/sliceruser/nnUNet_results"
imagesTrFolder= join(taskFolder,'imagesTr')
labelsTrFolder= join(taskFolder,'labelsTr')
imagesTsFolder= join(taskFolder,'imagesTs')
json_path= join(taskFolder,'dataset.json')

# main modality that will be set as a reference and others will be registered to it 


os.makedirs(nNunetBaseFolder ,exist_ok = True)
# os.makedirs(join(nNunetBaseFolder,'nnUNet_raw_data_base') ,exist_ok = True)
# os.makedirs(join(nNunetBaseFolder,'nnUNet_raw_data_base','nnUNet_raw_data') ,exist_ok = True)
os.makedirs(taskFolder ,exist_ok = True)
# os.makedirs(imagesTrFolder ,exist_ok = True)
# os.makedirs(labelsTrFolder ,exist_ok = True)
os.makedirs(preprocesss_folder ,exist_ok = True)
os.makedirs(results_folder ,exist_ok = True)
os.makedirs(mainResults_folder ,exist_ok = True)
os.makedirs(join(mainResults_folder,taskName),exist_ok = True)


# copy files from pretraining disk
# p = Popen(f"cp -a /home/sliceruser/preprocessed/anatomic_preprocessed/imagesTr {taskFolder}", shell=True)
# p.wait()
# p = Popen(f"cp -a /home/sliceruser/preprocessed/anatomic_preprocessed/labelsTr {taskFolder}", shell=True)
# p.wait()
# p = Popen(f"rm -r /home/sliceruser/preprocessed/anatomic_preprocessed/imagesTr ", shell=True)
# p.wait()
# p = Popen(f"rm -r /home/sliceruser/preprocessed/anatomic_preprocessed/labelsTr {taskFolder}", shell=True)
# p.wait()

os.environ["nnUNet_raw"] = "/home/sliceruser/preprocessed/anatomic_preprocessed"

channel_names={  
    "0": "t2w", 
    "1": "adc", 
    "2": "hbv", 
    }


label_names= {  
    "background": 0,
    "pz": 1,
    "tz": 2,
    "sv" :3,
    "full_prost":[1,2]
    }


data = { 
"channel_names": channel_names, 
"labels": label_names,  
"file_ending": ".nii.gz",
"overwrite_image_reader_writer": "SimpleITKIO",
"regions_class_order": [1,2,4,3],  
"normalization_schemes" : ["zscore","noNorm","noNorm"],
"numTraining" : len(os.listdir(labelsTrFolder)),
"nnUNetPlans" : ['2d','3d_lowres','3d_cascade_fullres', '3d_fullres']
}


json_string = json.dumps(data,sort_keys=False)
with open(json_path, 'w') as outfile:
    outfile.write(json_string)


cmd_terminal=f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
print(f"ccccc {cmd_terminal} \n")
p = Popen(cmd_terminal, shell=True)
p.wait()



# plans_path= join(preprocesss_folder,taskName,'nnUNetPlans.json')
plans_path= f"/home/sliceruser/nnunetMainFolder/nnUNet_preprocessed/Dataset{dataset_id}_Prostate/nnUNetPlans.json"
f = open(plans_path)
plans = json.load(f)
plans['configurations']['3d_lowres'] = {
    "data_identifier": "nnUNetPlans_3d_lowres",  # do not be a dumbo and forget this. I was a dumbo. And I paid dearly with ~10 min debugging time
    'inherits_from': '3d_fullres', 'preprocessor_name': 'DefaultPreprocessor'
        , 'batch_size': 2
        , "patch_size": [48, 192, 160]
        , "spacing": [3.299999952316284, 0.78125, 0.78125]
        , "UNet_base_num_features": 128
        , 'median_image_size_in_voxels': [ 42., 164., 159.]
        , 'normalization_schemes': ['NoNormalization', 'NoNormalization', 'NoNormalization']
        , 'use_mask_for_norm': [False, False, False], 'UNet_class_name': 'PlainConvUNet'
        #, 'UNet_base_num_features': 32
        , 'n_conv_per_stage_encoder': (2, 2, 2, 2, 2, 2), 'n_conv_per_stage_decoder': (2, 2, 2, 2, 2)
        , 'num_pool_per_axis': [3, 5, 5]
        , 'pool_op_kernel_sizes': [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        , 'conv_kernel_sizes': [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        , 'unet_max_num_features': 320, 'resampling_fn_data': 'resample_data_or_seg_to_shape'
        , 'resampling_fn_seg': 'resample_data_or_seg_to_shape'
        , 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}
        , 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}
        , 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape'
        , 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}
        , 'batch_dice': False}



json_string = json.dumps(plans,sort_keys=False)     
print(f"aaaaaaaaaaaaa {json_string} \n \n \n ppppppppppppppppppppppppppppppppppp")
with open(plans_path, 'w') as outfile:
    outfile.write(json_string)



cmd_terminal=f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
p = Popen(cmd_terminal, shell=True)
p.wait()

os.environ["nnUNet_raw"] ="/home/sliceruser/nnunetMainFolder/nnUNet_raw"