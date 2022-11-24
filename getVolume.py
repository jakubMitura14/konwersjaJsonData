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
import torch
from os import path as pathOs
import more_itertools
from os.path import basename, dirname, exists, isdir, join, split
import shutil
from mainFuncs import getLabelsAbbrev

def get_volume(colName,current_row, studyId, number, series_desc):
    """
    get volume of a binary mask that resides in path
    """
    path=current_row[colName]
    image3D=sitk.ReadImage(path)
    data=sitk.GetArrayFromImage(image3D)
    spacing = image3D.GetSpacing()
    #we get total vlume by multiplying sum of all voxels with each voxel volume
    volume = np.sum(data)*(spacing[0]*spacing[1]*spacing[2])/1000
    return (volume, studyId, number, series_desc,colName)

def getVolumes(current_row,col_names_to_volume):
    """
    given data of single series return the volume of each label 
    in case given label is not present return -1
    """
    current_row= current_row[1]
    #getting valid column names for this row
    col_names_for_volumes_inner = list(filter(  lambda colName :current_row[colName]!=" " , col_names_to_volume)) 
    #getting Paths
    volumesPaths = list(map(lambda colName:current_row[colName],col_names_for_volumes_inner))
    #calculate volumes of each label
    volumes= list(map(lambda colName:get_volume(colName,current_row ,current_row['study_id'],current_row['masterolds'],current_row['series_desc'] ), col_names_for_volumes_inner))

    # return list(zip(col_names_for_volumes_inner+rest_of_colnames,volumes+volumes_rest))
    return volumes

def get_volumes_frames(preprocessed_df,prost_volumes_csv_dir,lesion_volumes_csv_dir,prostateLabelName):
    """
    reads the data from associated data frame and creates new dataframe
    with data about volume of prostate
    with data about the volume of each lesion
    """
    
    #getting all lesion labels _noSeg
    cols=preprocessed_df.columns
    cols=list(filter(lambda el: 'lesion' in el ,cols))
    lesion_cols=list(filter(lambda el: '_noSeg' in el ,cols))
    

    prostate_volumes_data= list(map( partial(getVolumes,col_names_to_volume=[getLabelsAbbrev(prostateLabelName)+'_noSeg']), preprocessed_df.iterrows()))
    lesion_volumes_data= list(map( partial(getVolumes,col_names_to_volume=lesion_cols), preprocessed_df.iterrows()))
    prostate_volumes_data=list(itertools.chain(*prostate_volumes_data))

    prostate_volumes_frame= pd.DataFrame()
    lesion_volumes_frame= pd.DataFrame()
    #populating frame with data
    prostate_volumes_frame['study_id']=list(map(lambda volEl : volEl[1]  ,prostate_volumes_data))
    prostate_volumes_frame['number']=list(map(lambda volEl : volEl[2]  ,prostate_volumes_data))
    prostate_volumes_frame['prostate_volume']=list(map(lambda volEl : volEl[0]  ,prostate_volumes_data))

    #populating frame with data
    lesion_volumes_data=list(itertools.chain(*lesion_volumes_data))


    #volume, studyId, number, series_desc,colName

    lesion_volumes_frame['study_id']=list(map(lambda el: el[1],lesion_volumes_data )) 
    lesion_volumes_frame['number']=list(map(lambda el: el[2],lesion_volumes_data )) 
    lesion_volumes_frame['modality']=list(map(lambda el: el[3],lesion_volumes_data )) 
    lesion_volumes_frame['volume']=list(map(lambda el: el[0],lesion_volumes_data )) 
    lesion_volumes_frame['lesion_name']=list(map(lambda el: el[4].split('_')[0],lesion_volumes_data )) 
    lesion_volumes_frame['annotator_id']=list(map(lambda el: f"{el[4].split('_')[1]}_{el[4].split('_')[2]}",lesion_volumes_data )) 

    
    # for col_vol_name in col_names_to_volume:
    #     curr_vol_dat= list(map(lambda zipped : list(filter(lambda tupl: tupl[0]==col_vol_name ,zipped ))[0]  ,all_volumes_data))
    #     volumes_frame[col_vol_name]=curr_vol_dat
    
    print(f"savinggg {prostate_volumes_frame}")
    prostate_volumes_frame.to_csv(prost_volumes_csv_dir) 
    lesion_volumes_frame.to_csv(lesion_volumes_csv_dir) 

    return (prostate_volumes_frame,lesion_volumes_frame) 

# cols=out_files_frame.columns
# cols=list(filter(lambda el: 'lesion' in el ,cols))
# cols=list(filter(lambda el: '_noSeg' in el ,cols))
# cols