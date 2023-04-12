
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
import mainFuncs
import itertools
from pydicom.fileset import FileSet
from os import path as pathOs
from pathlib import Path
import toolz
from toolz.curried import pipe, map, filter, get
from toolz import curry
import getDirAndnumbFrame
from getDirAndnumbFrame import get_df_orig_dir_info
import get3dFiles
from get3dFiles import get_frame_with_output
import preprocess
from preprocess import dilatate_erode_conditionally
from os.path import basename, dirname, exists, isdir, join, split
import nnunetv2

import elastixRegister
from elastixRegister import reg_a_to_b
import tempfile
import shutil
import re
from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import SimpleITK as sitk



file_path='/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset279_Prostate/imagesTr/9002600_0000.nii.gz'
file_path_b='/workspaces/konwersjaJsonData/explore/9004900_0000.nii.gz'
file_path_orig= '/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont/Data/001/1_t2w.mha'
image = sitk.ReadImage(file_path)
print(image)
# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset279_Prostate/imagesTr/9004900_0000.nii.gz /home/sliceruser/explore/9004900_0000.nii.gz

