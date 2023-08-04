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
from map_anatomy_on_adc import save_mean_anatomy_adc
from get_max_perimeter import save_max_paerimeter

#JSON filr from mdai summarizing data about annotations etc. 
JSON = '/home/sliceruser/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_2023-07-18-083817.json'
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
jsonFolder='/home/sliceruser/code/dicomSeg'
#csv witch indicates what lesions should not be included and what should be the lesions numbering
correctionsCSVDir= '/home/sliceruser/workspaces/konwersjaJsonData/metaDataProject/valid_lesion_mapping.csv'
neededNumbersCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/metaDataProject/AI4AR_PCa_radiological.csv'# 'AI4A4_PCa_clinical.csv' #frame where we will get those ids that we are intrested in
#loading data from JSON
results = mdai.common_utils.json_to_dataframe(JSON)
results['annotations'].to_csv('/workspaces/konwersjaJsonData/explore/mdai_prim_results.csv') 
annot=results['annotations']
#load data about files in directory folder
#files passed from the folder where they were downloaded by mdai client
files_df= mainFuncs.get_df_file_info(dataDir,client_down_csv)
#adding data  about original folders from orig folder
# files_df_origFolds= get_df_orig_dir_info(testXnatPath,orig_data_csv)
#add data about original folder structure to frame based on data downloaded with mdai client
# files_df_with_orig_folds=getDirAndnumbFrame.add_orig_dir_data(files_df, files_df_origFolds)

os.makedirs('/home/sliceruser/workspaces/konwersjaJsonData/outCsv' ,exist_ok = True)
os.makedirs(outputDir ,exist_ok = True)
################
#just for debuging
# files_df_with_orig_folds=files_df_origFolds.loc[files_df_origFolds['masterolds'] == 930]
# files_df_origFolds=files_df_origFolds.loc[files_df_origFolds['masterolds'] == 930]
###############

#parsing files and saving 3D data in the output folder
out_files_frame= get_frame_with_output(files_df,annot,outputDir,resCSVDir,mainFoldDirMha,mainFoldDirSeg,jsonFolder,correctionsCSVDir,neededNumbersCSVDir)

### preprocessing
#we first define the labels that should as a sum be included in a prostate gland 
# so we get rid of overlaps and inconsistencies with main prostate mask
prostateLab = 'prostate'
#files will be overwritten in the output folder
labelsOfIntrest = ['peripheral zone',  'transition zone','anterior fibromuscular stroma', 'central zone', 'urethra','seminal vesicles L',
       'seminal vesicles R']#
####additionally dilatate_erode_conditionally after processing saves dicom seg's into previosly created folders
#dilatate_erode_conditionally(out_files_frame,labelsOfIntrest,prostateLab,annot,jsonFolder)  #TODO(unhash)

##measurements
#after data is preprocessed we will perform measurements like volumes of labels

#directory of the csv to save data about volumes
prost_volumes_csv_dir='/workspaces/konwersjaJsonData/explore/prost_volumes.csv'
lesion_volumes_csv_dir='/workspaces/konwersjaJsonData/explore/lesion_volumes.csv'
adc_means_csv_dir='/workspaces/konwersjaJsonData/explore/mean_adcs.csv'
new_anatomy_csv_dir='/workspaces/konwersjaJsonData/explore/new_anatomies.csv'

anatomic_cols=['pg_noSeg','afs_noSeg','cz_noSeg','pz_noSeg','sv_l_noSeg','sv_r_noSeg','tz_noSeg','ur_noSeg']
# get_volumes_frames(out_files_frame,prost_volumes_csv_dir,lesion_volumes_csv_dir,prostateLab,anatomic_cols,adc_means_csv_dir)
# measurements.get_new_anatomu_inter_observer_agreement(out_files_frame,new_anatomy_csv_dir)

anatomy_adc_csv_dir='/workspaces/konwersjaJsonData/explore/mean_adcs_per_anatomy.csv'
save_mean_anatomy_adc(out_files_frame,anatomic_cols,anatomy_adc_csv_dir)
# circ_frame_csv_dir='/workspaces/konwersjaJsonData/explore/max_circs.csv'
# save_max_paerimeter(out_files_frame,anatomic_cols,circ_frame_csv_dir)


# sudo cp -a /home/jm/Downloads/to_eval_test_cases /media/jm/hddData/datasets/workspace/konwersjaJsonData/to_eval_test_cases


# cp -a /tmp/tmpla39qr7t /workspaces/konwersjaJsonData/tmp


