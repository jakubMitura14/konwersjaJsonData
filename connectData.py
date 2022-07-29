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
import getDirAndnumbFrame
from getDirAndnumbFrame import get_df_orig_dir_info
import get3dFiles
from get3dFiles import get_frame_with_output
import preprocess
from preprocess import dilatate_erode_conditionally
from os.path import basename, dirname, exists, isdir, join, split
import measurements

#JSON filr from mdai summarizing data about annotations etc. 
JSON = '/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_dataset_D_gQm1nQ_2022-07-15-104055.json'

#directory where the output will be stored
outputDir='/workspaces/konwersjaJsonData/output'
#directory of the csv storing most relevant data per each series
resCSVDir='/workspaces/konwersjaJsonData/resCSV'
#directory of the csv to save data about dice scores of compared lesions
dice_csv_dir='/workspaces/konwersjaJsonData/diceFrame.csv'
volumes_csv_dir='/workspaces/konwersjaJsonData/volumesFrame.csv'
#downloaded manually
orig_data_dir='/workspaces/konwersjaJsonData/nas-lssi-dco'
orig_data_csv='/workspaces/konwersjaJsonData/orig_files.csv'
#path to folder with all required data downloaded with mdai client
dataDir='/workspaces/konwersjaJsonData/data'
client_down_csv='/workspaces/konwersjaJsonData/client_files.csv'


#loading data from JSON
results = mdai.common_utils.json_to_dataframe(JSON)
annot=results['annotations']

#load data about files in directory folder

#files passed from the folder where they were downloaded by mdai client
files_df= mainFuncs.get_df_file_info(dataDir,client_down_csv)
#adding data  about original folders from orig folder
files_df_origFolds= get_df_orig_dir_info(orig_data_dir,orig_data_csv)
#add data about original folder structure to frame based on data downloaded with mdai client
files_df_with_orig_folds=getDirAndnumbFrame.add_orig_dir_data(files_df, files_df_origFolds)

#parsing files and saving 3D data in the output folder
out_files_frame= get_frame_with_output(files_df_with_orig_folds,annot,outputDir,resCSVDir)


### preprocessing
#we first define the labels that should as a sum be included in a prostate gland 
# so we get rid of overlaps and inconsistencies with main prostate mask
prostateLab = 'prostate'
labelsOfIntrest = ['peripheral zone',  'transition zone','anterior fibromuscular stroma', 'central zone', 'urethra']
#files will be overwritten in the output folder
#dilatate_erode_conditionally(out_files_frame,labelsOfIntrest,prostateLab,annot)  #TODO(unhash)

##measurements
#after data is preprocessed we will perform measurements like volumes of labels
#names of labels in which we will measure volumes
col_names_to_volume = ['anterior fibromuscular stroma', 'central zone', 
       'external iliac', 'internal iliac', 'lesion 1', 'lesion 2', 'lesion 3',
       'lesion 4', 'lymph node regional', 'lymph node regional group',
       'obturator', 'peripheral zone', 'prostate', 'seminal vesicles L',
       'seminal vesicles R', 'transition zone', 'urethra']

#names of labels that will be compare between themselves between diffrent annotators and DICE score will be saved
col_names_for_dice=['lesion 1', 'lesion 2', 'lesion 3','lesion 4']

all_volumes_data,dice_df=measurements.get_volume_and_dice_data(col_names_for_dice,col_names_to_volume, out_files_frame,volumes_csv_dir,dice_csv_dir)


