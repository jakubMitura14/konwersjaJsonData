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

### first one need to invoke the get3dFiles to produce dataframe and save it in resCSVDir
resCSVDir='/workspaces/konwersjaJsonData/resCSV'
resCSVMeasuresDir='/workspaces/konwersjaJsonData/resCSVMeasures.csv'

JSON = '/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_dataset_D_gQm1nQ_2022-07-15-104055.json'
results = mdai.common_utils.json_to_dataframe(JSON)
#so we have dictionary of dataframes
results.keys()#'annotations', 'studies', 'labels'
annot=results['annotations']

files_df=pd.read_csv(resCSVDir)


col_names_to_volume = ['anterior fibromuscular stroma', 'central zone', 'curvilinear contact',
       'external iliac', 'internal iliac', 'lesion 1', 'lesion 2', 'lesion 3',
       'lesion 4', 'lymph node regional', 'lymph node regional group',
       'obturator', 'peripheral zone', 'prostate', 'seminal vesicles L',
       'seminal vesicles R', 'transition zone', 'urethra']

col_names_for_dice=['lesion 1', 'lesion 2', 'lesion 3','lesion 4']

for_dice_all_pairs=list(more_itertools.powerset(col_names_for_dice))
for_dice_all_pairs=list(filter(lambda tupl:len(tupl)==2  ,for_dice_all_pairs))



def get_volume(path):
    """
    get volume of a binary mask that resides in path
    """
    image3D=sitk.ReadImage(path)
    data=sitk.GetArrayFromImage(image3D)
    spacing = image3D.GetSpacing()
    #we get total vlume by multiplying sum of all voxels with each voxel volume
    volume = np.sum(data)*( spacing[0]*spacing[1]*spacing[2]  )
    return volume

def get_dice_between(pathA,pathB):
    """
    from http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    get two images and return dice coefficient
    """
    imageA=sitk.ReadImage(pathA)
    imageB=sitk.ReadImage(pathB)
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(imageA, imageB)
    return overlap_measures_filter.GetDiceCoefficient()




def getVolumes(current_row):
    current_row= current_row[1]
    #getting valid column names for this row
    col_names_for_volumes_inner = list(filter(  lambda colName :current_row[colName]!=" " , col_names_to_volume)) 
    #getting Paths
    volumesPaths = list(map(lambda colName:current_row[colName],col_names_for_volumes_inner))
    #calculate volumes of each label
    volumes= list(map(get_volume, volumesPaths  ))
    return list(zip(col_names_for_volumes_inner,volumes))

list_volumes=list(map( getVolumes  ,list(files_df.iterrows())))
#flattening
list_volumes=list(itertools.chain(*list_volumes))





series = np.unique(files_df['series_id'].to_numpy())
currentSeries = series[0]
locDf = files_df.loc[files_df['series_id'] == currentSeries]
locDf
# colPairs_for_Dice = list(filter(  lambda tupl : (current_row[tupl[0]]!=" " and current_row[tupl[1]]!=" " ), for_dice_all_pairs)) 

# forDicePaths= list(map(lambda tupl: (current_row[tupl[0]],current_row[tupl[1]]),colPairs_for_Dice))


files_df.columns


studies = np.unique(files_df['study_id'].to_numpy())
currentStudy = studies[0]
locDf = files_df.loc[files_df['study_id'] == currentStudy]
loc_series = np.unique(locDf['series_id'].to_numpy())


series = np.unique(files_df['series_id'].to_numpy())
currentSeries = series[31]
locDf = files_df.loc[files_df['series_id'] == currentSeries]
locDf

currentSeries = loc_series[1]
series_df = files_df.loc[files_df['series_id'] == currentSeries]

series_doctors= np.unique(series_df['doctor_id'].to_numpy())


locDf['series_id'].to_numpy()

series_df.columns

