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

from os import path as pathOs

def get_df_orig_dir_info(dataDir):
    listOfPatsh=get_all_file_paths(dataDir)
    df = pd.DataFrame()
    df['paths']=listOfPatsh
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(getUids  ,list(df.iterrows()))

    sops = list(map(lambda tupl: tupl[0], resList))
    SeriesInstanceUIDs = list(map(lambda tupl: tupl[1], resList))
    InstanceUIDs = list(map(lambda tupl: tupl[2], resList))
    df['SOPInstanceUID']=sops   
    df['SeriesInstanceUID']=SeriesInstanceUIDs   
    df['StudyInstanceUID']=InstanceUIDs   
    return df 