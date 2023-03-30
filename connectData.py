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
import measurements
import re
import shutil
import getVolume
from getVolume import get_volumes_frames

#JSON filr from mdai summarizing data about annotations etc. 
JSON = '/home/sliceruser/workspaces/konwersjaJsonData/out/mdai_public_project_gaq3y0Rl_annotations_2023-03-28-155602.json'
#directory where the output will be stored
outputDir='/home/sliceruser/workspaces/konwersjaJsonData/output'
#directory of the csv storing most relevant data per each series
resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
#directory of the csv to save data about volumes
prost_volumes_csv_dir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/prost_volumes.csv'
lesion_volumes_csv_dir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/lesion_volumes.csv'
#downloaded manually
# orig_data_dir='//home/sliceruser/workspaces/konwersjaJsonData/nas-lssi-dco'
testXnatPath= '/home/sliceruser/workspaces/konwersjaJsonData/TEST'
orig_data_csv='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/orig_files.csv'
#path to folder with all required data downloaded with mdai client
dataDir='/home/sliceruser/workspaces/konwersjaJsonData/data'
client_down_csv='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/client_files.csv'
#name of the folder where files with comparisons of lesions will be stored
rootFolder_lesion_analysis='/home/sliceruser/workspaces/konwersjaJsonData/forLesionAnalysis'
mainFoldDirMha='/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont'
mainFoldDirSeg='/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_dicom'
#folder with definitions for dicom segs
jsonFolder='/home/sliceruser/workspaces/konwersjaJsonData/dicomSeg'
#csv witch indicates what lesions should not be included and what should be the lesions numbering
correctionsCSVDir= '/home/sliceruser/workspaces/konwersjaJsonData/parsedd.csv'
neededNumbersCSVDir= 'AI4A4_PCa_clinical.csv' #frame where we will get those ids that we are intrested in
#loading data from JSON
results = mdai.common_utils.json_to_dataframe(JSON)
annot=results['annotations']
#load data about files in directory folder
#files passed from the folder where they were downloaded by mdai client
files_df= mainFuncs.get_df_file_info(dataDir,client_down_csv)
#adding data  about original folders from orig folder
files_df_origFolds= get_df_orig_dir_info(testXnatPath,orig_data_csv)
#add data about original folder structure to frame based on data downloaded with mdai client
files_df_with_orig_folds=getDirAndnumbFrame.add_orig_dir_data(files_df, files_df_origFolds)

################
#just for debuging
# files_df_with_orig_folds=files_df_origFolds.loc[files_df_origFolds['masterolds'] == 930]
# files_df_origFolds=files_df_origFolds.loc[files_df_origFolds['masterolds'] == 930]
###############

#parsing files and saving 3D data in the output folder
out_files_frame= get_frame_with_output(files_df_with_orig_folds,files_df_origFolds,annot,outputDir,resCSVDir,mainFoldDirMha,mainFoldDirSeg,jsonFolder,correctionsCSVDir,neededNumbersCSVDir)

### preprocessing
#we first define the labels that should as a sum be included in a prostate gland 
# so we get rid of overlaps and inconsistencies with main prostate mask
prostateLab = 'prostate'
#files will be overwritten in the output folder
labelsOfIntrest = ['peripheral zone',  'transition zone','anterior fibromuscular stroma', 'central zone', 'urethra','seminal vesicles L',
       'seminal vesicles R']#
####additionally dilatate_erode_conditionally after processing saves dicom seg's into previosly created folders
dilatate_erode_conditionally(out_files_frame,labelsOfIntrest,prostateLab,annot,jsonFolder)  #TODO(unhash)

##measurements
#after data is preprocessed we will perform measurements like volumes of labels
get_volumes_frames(out_files_frame,prost_volumes_csv_dir,lesion_volumes_csv_dir,prostateLab)



# df[df.A == 'foo']

# neededNumbersCSV=pd.read_csv(neededNumbersCSVDir)
# neededIds= np.unique(neededNumbersCSV['patient_id'].to_numpy())
# neededIds = list(map( lambda el: int(el),neededIds))
# len(neededIds)

# bb=pd.read_csv(resCSVDir)
# neededIdsB= np.unique(bb['masterolds'].to_numpy())
# len(neededIdsB)
# resCSVDir

# neededIdsC= np.unique(files_df_origFolds['masterolds'].to_numpy())
# len(neededIdsC)

# neededIdsC= list(map(lambda el: int(el),neededIdsC))

# yy = list(files_df_origFolds['masterolds'].to_numpy())
# yy= list(map(lambda el: int(el) in neededIds,yy))
# yy

# dd=files_df_origFolds.loc[yy]
# dd


# ff = list(dd['masterolds'].to_numpy())
# ff= np.unique(ff)
# len(ff)




# ee= list(filter(lambda el: el in neededIds ,neededIdsC))
# len(ee)

# files_df_origFolds

# import os
# path = "//home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont/Data"


# for dir, sub_dirs, files in os.walk(path):
#     print(len(sub_dirs))

# files = [x[0] + "/" + y for x in os.walk(path) if len(x[-1]) > 0 for y in x[-1]]
# len(files)

## error 140,930 - can not find it in correction df although is in source df's




# "name":"Dataset Test"

# ##measurements
# #after data is preprocessed we will perform measurements like volumes of labels
# #names of labels in which we will measure volumes
# col_names_to_volume = ['anterior fibromuscular stroma', 'central zone', 
#        'external iliac', 'internal iliac', 'lesion 1', 'lesion 2', 'lesion 3',
#        'lesion 4','lesion 5', 'lymph node regional', 'lymph node regional group',
    #    'obturator', 'peripheral zone', 'prostate', 'seminal vesicles L',
    #    'seminal vesicles R', 'transition zone', 'urethra']

# #names of labels that will be compare between themselves between diffrent annotators and DICE score will be saved
# col_names_for_dice=['lesion 1', 'lesion 2', 'lesion 3','lesion 4']

# all_volumes_data,dice_df=measurements.get_volume_and_dice_data(col_names_for_dice,col_names_to_volume, out_files_frame,volumes_csv_dir,dice_csv_dir)

# #saving the lesions and their common parts
# measurements.save_lesions_consensus(dice_df,rootFolder_lesion_analysis )

# def copyTo(row, nameBeg, targetDir,sourceRow='series_MRI_path'):
#        row=row[1]
#        num= row["masterolds"]
#        source=row[sourceRow]
#        destination = join(targetDir,f"{num}_{nameBeg}.mha")
#        shutil.copy(source, destination)

# t2wDf =out_files_frame.loc[out_files_frame['series_desc'] == 't2_transverse']
# adcDf =out_files_frame.loc[out_files_frame['series_desc'] == 'adc_transverse']
# hbvDf =out_files_frame.loc[out_files_frame['series_desc'] == 'dwi_transverse']

# list(map( partial(copyTo, nameBeg='t2w',targetDir='//home/sliceruser/workspaces/konwersjaJsonData/forTestPicai/t2w' ),list(t2wDf.iterrows())))
# list(map( partial(copyTo, nameBeg='adc',targetDir='//home/sliceruser/workspaces/konwersjaJsonData/forTestPicai/adc' ),list(adcDf.iterrows())))
# list(map( partial(copyTo, nameBeg='hbv',targetDir='//home/sliceruser/workspaces/konwersjaJsonData/forTestPicai/hbv' ),list(hbvDf.iterrows())))

# t2wPaths= np.unique(t2wDf['series_MRI_path'].to_numpy())
# prostateDf = out_files_frame.loc[out_files_frame['prostate'] != ' ']
# list(map( partial(copyTo, nameBeg='t2w',targetDir='//home/sliceruser/workspaces/konwersjaJsonData/forTestPicai/prostatefull/prostateIm' ),list(prostateDf.iterrows())))



# out_files_frame.columns









##changing names of the directories from series number to folder number in original folders 

#import re
#https://www.tutorialspoint.com/python/os_rename.htm
#os.rename(src, dst)





# import preprocess

# # dilatate_erode_conditionally(out_files_frame,labelsOfIntrest,prostateLab,annot)  #TODO(unhash)

# #usefull to iterate around not including iteration in z direction
# indicies_around=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([0])))
# indicies_around_full=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([-1,0,1])))

# # using only those rows where we have prostate
# frame_of_intr=out_files_frame.loc[out_files_frame[prostateLab]!=" "]
# current_row=list(frame_of_intr.iterrows())[11]

# labelsOfIntrest = ['peripheral zone',  'transition zone','anterior fibromuscular stroma', 'central zone', 'urethra']

# grow_labels(current_row=current_row,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab,indicies_around_full=indicies_around_full)


# current_row[1]['prostate']

# # '//home/sliceruser/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.8.15.100960.30000022021714463775000004282/U_8ZmM76/1.3.12.2.1107.5.8.15.100960.30000022021714463775000004504/prostate.nii.gz'