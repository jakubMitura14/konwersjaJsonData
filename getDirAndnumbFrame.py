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

orig_data_dir='/workspaces/konwersjaJsonData/nas-lssi-dco'

#getting paths to dicom dir files
listOfPatsh=mainFuncs.get_all_file_paths(orig_data_dir)
pathsDICOMDIR= list(filter(lambda pathh:  'DICOMDIR' in pathh,listOfPatsh ))


currentDicomDirPath=pathsDICOMDIR[0]
pathhh = Path(currentDicomDirPath)
currentFold=str(os.path.relpath(pathhh.parent))

ds = pydicom.dcmread(currentDicomDirPath)
fs = FileSet(ds)
listInfs = list(fs)
current_file_ref = listInfs[0]

def getFileData(current_file,currentFold)
    current_file = current_file_ref.load()
    sop=mainFuncs.get_SOPInstanceUID(current_file)
    SeriesInstanceUID= mainFuncs.get_SeriesInstanceUID(current_file)
    stidy_instanceUID=mainFuncs.get_StudyInstanceUID(current_file)
    return (currentFold,stidy_instanceUID,SeriesInstanceUID,sop  )




exampleDicomDir= '/workspaces/konwersjaJsonData/nas-lssi-dco/829/DICOMDIR'
ds = pydicom.dcmread(exampleDicomDir)
fs = FileSet(ds)

for instance in fs:
    print(instance.load().file_meta)
    

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