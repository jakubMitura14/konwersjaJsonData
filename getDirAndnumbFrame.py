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

def getListMeta(currentDicomDirPath):
    """
    iterates ove files associated with single dicom dir
    and returns the metadata related to them
    """
    pathhh = Path(currentDicomDirPath)
    currentFold=str(os.path.relpath(pathhh.parent))
    ds = pydicom.dcmread(currentDicomDirPath)
    fs = FileSet(ds)
    fileMataDatas=list(map(partial(getFileData,currentFold=currentFold),list(fs)))
    return fileMataDatas

def getFileData(current_file_ref,currentFold):
    """
    given file return its series, sop and study UID
    """
    filePath=current_file_ref.path
    current_file = current_file_ref.load()
    sop=mainFuncs.get_SOPInstanceUID(current_file)
    SeriesInstanceUID= mainFuncs.get_SeriesInstanceUID(current_file)
    stidy_instanceUID=mainFuncs.get_StudyInstanceUID(current_file)
    return (currentFold,filePath,stidy_instanceUID,SeriesInstanceUID,sop  )


def get_df_orig_dir_info(orig_data_dir,csvDir):
    """
    iterates over files and saves ther paths and matadata to dataframe and subsequently to csv
    """
    if(pathOs.exists(csvDir)):
        return pd.read_csv(csvDir)    
    #get all paths of DICOMDIR files
    listOfPatsh=mainFuncs.get_all_file_paths(orig_data_dir)
    pathsDICOMDIR= list(filter(lambda pathh:  'DICOMDIR' in pathh,listOfPatsh ))
    df = pd.DataFrame()
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(getListMeta ,pathsDICOMDIR)
    #flatten array
    resList=list(itertools.chain(*resList))

    masterolds=list(map(lambda tupl: tupl[0], resList))
    currentFilePath=list(map(lambda tupl: tupl[1], resList))
    StudyInstanceUID = list(map(lambda tupl: tupl[2], resList))
    SeriesInstanceUIDs = list(map(lambda tupl: tupl[3], resList))
    sops = list(map(lambda tupl: tupl[4], resList))

    df['masterolds']=masterolds   
    df['paths']=currentFilePath   
    df['SOPInstanceUID']=sops   
    df['SeriesInstanceUID']=SeriesInstanceUIDs   
    df['StudyInstanceUID']=StudyInstanceUID  
    df.to_csv(csvDir) 
    return df 

def get_orig_fold(uid,dictt):
    """
    getting the folder name in original folder
    """
    if(uid in dictt):
            return dictt[uid]
    return " "       

def add_orig_dir_data(files_df, files_df_origFolds):
       """
       we got the data about original folder s from orig folders frame
       now we add this data to data downloaded from mdai by client library
       """
       smaller_df=files_df_origFolds[['masterolds','StudyInstanceUID']].drop_duplicates()
       masterolds=smaller_df['masterolds'].to_numpy()
       StudyInstanceUID=smaller_df['StudyInstanceUID'].to_numpy()
       dictt ={StudyInstanceUID[i]: masterolds[i] for i in range(len(masterolds))}
       StudyInstanceUID_mdai= files_df['StudyInstanceUID'].to_numpy()
       orig_folds_dat= list(map(lambda uid :get_orig_fold(uid,dictt) ,StudyInstanceUID_mdai))
       files_df['masterolds']=orig_folds_dat
       return files_df




    