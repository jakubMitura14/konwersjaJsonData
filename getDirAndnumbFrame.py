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
resCSVDir='/workspaces/orig_files_dir.csv'
#getting paths to dicom dir files




def getListMeta(currentDicomDirPath):
    pathhh = Path(currentDicomDirPath)
    currentFold=str(os.path.relpath(pathhh.parent))
    ds = pydicom.dcmread(currentDicomDirPath)
    fs = FileSet(ds)
    fileMataDatas=list(map(partial(getFileData,currentFold=currentFold),list(fs)))
    return fileMataDatas

def getFileData(current_file_ref,currentFold):
    filePath=current_file_ref.path
    current_file = current_file_ref.load()
    sop=mainFuncs.get_SOPInstanceUID(current_file)
    SeriesInstanceUID= mainFuncs.get_SeriesInstanceUID(current_file)
    stidy_instanceUID=mainFuncs.get_StudyInstanceUID(current_file)
    return (currentFold,filePath,stidy_instanceUID,SeriesInstanceUID,sop  )


def get_df_orig_dir_info(orig_data_dir):
    #get all paths of DICOMDIR files
    listOfPatsh=mainFuncs.get_all_file_paths(orig_data_dir)
    pathsDICOMDIR= list(filter(lambda pathh:  'DICOMDIR' in pathh,listOfPatsh ))
    df = pd.DataFrame()
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(getListMeta ,pathsDICOMDIR)
    #flatten array
    resList=list(itertools.chain(*resList))

# (currentFold,filePath,stidy_instanceUID,SeriesInstanceUID,sop  )
    masterolds=list(map(lambda tupl: tupl[0], resList))
    currentFilePath=list(map(lambda tupl: tupl[1], resList))
    StudyInstanceUID = list(map(lambda tupl: tupl[2], resList))
    SeriesInstanceUIDs = list(map(lambda tupl: tupl[3], resList))
    sops = list(map(lambda tupl: tupl[4], resList))

    df['masterolds']=masterolds   
    df['currentFilePath']=currentFilePath   
    df['SOPInstanceUID']=sops   
    df['SeriesInstanceUID']=SeriesInstanceUIDs   
    df['StudyInstanceUID']=StudyInstanceUID   
    return df 


resDf=get_df_orig_dir_info(orig_data_dir)
resDf.to_csv(resCSVDir)

print(resDf)

# currentDicomDirPath=pathsDICOMDIR[0]

# allMetaDat=[]
# with mp.Pool(processes = mp.cpu_count()) as pool:
#     allMetaDat=pool.map( getListMeta ,pathsDICOMDIR )

# allMetaDat_flat=list(itertools.chain(*allMetaDat))

# print(allMetaDat_flat)


    