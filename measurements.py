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

### first one need to invoke the get3dFiles to produce dataframe and save it in resCSVDir
# resCSVDir='/workspaces/konwersjaJsonData/resCSV'
# resCSVMeasuresDir='/workspaces/konwersjaJsonData/resCSVMeasures.csv'

# JSON = '/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_dataset_D_gQm1nQ_2022-07-15-104055.json'
# results = mdai.common_utils.json_to_dataframe(JSON)
# #so we have dictionary of dataframes
# results.keys()#'annotations', 'studies', 'labels'
# annot=results['annotations']

# files_df=pd.read_csv(resCSVDir)


# col_names_to_volume = ['anterior fibromuscular stroma', 'central zone', 'curvilinear contact',
#        'external iliac', 'internal iliac', 'lesion 1', 'lesion 2', 'lesion 3',
#        'lesion 4', 'lymph node regional', 'lymph node regional group',
#        'obturator', 'peripheral zone', 'prostate', 'seminal vesicles L',
#        'seminal vesicles R', 'transition zone', 'urethra']

# col_names_for_dice=['lesion 1', 'lesion 2', 'lesion 3','lesion 4']


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




def getVolumes(current_row,col_names_to_volume):
    current_row= current_row[1]
    #getting valid column names for this row
    col_names_for_volumes_inner = list(filter(  lambda colName :current_row[colName]!=" " , col_names_to_volume)) 
    rest_of_colnames= list(filter(  lambda colName :current_row[colName]==" " , col_names_to_volume))
    #getting Paths
    volumesPaths = list(map(lambda colName:current_row[colName],col_names_for_volumes_inner))
    #calculate volumes of each label
    volumes= list(map(get_volume, volumesPaths))
    #dummy tensor with minus 1 to keepshape consistent
    volumes_rest=  list(map(lambda x:-1 ,rest_of_colnames))
    return list(zip(col_names_for_volumes_inner+rest_of_colnames,volumes+volumes_rest))



def get_label_per_doc(doctor_id,locDf,col_names_for_dice):
    locLocDf = list(locDf.loc[locDf['doctor_id'] == doctor_id].iterrows())[0][1]
    liist= list( filter(lambda clN: locLocDf[clN]!=" " ,col_names_for_dice))
    return list(map(lambda entry : doctor_id+'_@_'+entry   ,liist))



def get_dice_from_tupl(tupl, locDf):
    changeA= tupl[0].split('_@_')
    changeB= tupl[1].split('_@_')
    
    doctorA= changeA[0]
    doctorB= changeB[0]

    labelA = changeA[1]
    labelB = changeB[1]

    pathA = list(locDf.loc[locDf['doctor_id'] == doctorA].iterrows())[0][1][labelA]
    pathB = list(locDf.loc[locDf['doctor_id'] == doctorB].iterrows())[0][1][labelB]
    return get_dice_between(pathA,pathB)



def get_dice_in_all_pairs(currentSeries,preprocessed_df,col_names_for_dice):
    """
    analyzing all possible combination o lesions descriped by diffrent annotators
    calculating dice score between them and returning nested tuple with series id and dice score for each pair
    """
    
    locDf = preprocessed_df.loc[preprocessed_df['series_id'] == currentSeries]
    doctors= np.unique(locDf['doctor_id'].to_numpy())
    if(len(doctors)>1):
        #getting the combinations of possible labels and doctor comparing all
        labels_per_doctor =list( map(partial(get_label_per_doc, col_names_for_dice=col_names_for_dice,locDf=locDf ),doctors   ))
        labels_per_doctor=list(itertools.chain(*labels_per_doctor))

#        print(labels_per_doctor)

        labels_per_doctor_all_pairs=list(more_itertools.powerset(labels_per_doctor))
        #filter only those of length 2
        labels_per_doctor_all_pairs=list(filter(lambda tupl:len(tupl)==2  ,labels_per_doctor_all_pairs))
        #filter only those that has in single tuple diffrent doctors
        labels_per_doctor_all_pairs=list(filter(lambda tupl: tupl[0].split('_@_')[0]!=tupl[1].split('_@_')[0]  ,labels_per_doctor_all_pairs))
        #calculating dice values
        dice_vals= list(map(partial(get_dice_from_tupl,locDf=locDf), labels_per_doctor_all_pairs ))
        #adding data about series to each tuple
        labels_per_doctor_all_pairs= list(map( lambda tupl: (currentSeries,tupl[0],tupl[1] )   ,labels_per_doctor_all_pairs))
        labels_per_doctor_all_pairs= list(zip(labels_per_doctor_all_pairs,dice_vals))
        #prepared for 
        labels_per_doctor_all_pairs=list(map(lambda tupl : (tupl[0][0],tupl[0][1].split('_@_')[0],tupl[0][1].split('_@_')[1]
                                  ,tupl[0][2].split('_@_')[0],tupl[0][2].split('_@_')[1] ,tupl[1])   ,labels_per_doctor_all_pairs))
        return labels_per_doctor_all_pairs
    return [(" ", " ", " ", " ", " ", " ")]

def get_volume_and_dice_data(col_names_for_dice,col_names_to_volume, preprocessed_df,volumes_csv_dir,dice_csv_dir):
    """
    reads the data from associated data frame and creates new dataframe
    with data about volumes 
    with data about the volume of each label where we want it
    additionally creates a new dataframe with data about dice scores
    """
    all_volumes_data= list(map( partial(getVolumes,col_names_to_volume=col_names_to_volume)   , preprocessed_df.iterrows()))
    volumes_frame= pd.DataFrame()
    #populating frame with data
    volumes_frame['study_id']=preprocessed_df['study_id']   
    volumes_frame['doctor_id']=preprocessed_df['doctor_id']   
    volumes_frame['series_id']=preprocessed_df['series_id']
    
    for col_vol_name in col_names_to_volume:
        curr_vol_dat= list(map(lambda zipped : list(filter(lambda tupl: tupl[0]==col_vol_name ,zipped ))[0]  ,all_volumes_data))
        volumes_frame[col_vol_name]=curr_vol_dat
    
    #### Dice
    series = np.unique(preprocessed_df['series_id'].to_numpy())
    list_dice_score = list(map( partial(get_dice_in_all_pairs,preprocessed_df=preprocessed_df,col_names_for_dice=col_names_for_dice),series ))
    list_dice_score=list(itertools.chain(*list_dice_score))

    #(currentSeries,list(zip(labels_per_doctor_all_pairs,dice_vals)))
    #so now we have the series of nested tuples entry 1 is series uid second entry is list of tuples where fitst entry marks what
    #lesion and which annotator did it
    # goal isto  create dataframe where column
        #1) series id 2) first annotator id 3) first lesion name 4) second annotator id 5) socond lesion name 6) dice score
    dice_df = pd.DataFrame(list_dice_score, columns =['SeriesId', 'doctor_a', 'lesion_a','doctor_b','lesion_b', 'dice'])
    volumes_frame.to_csv(volumes_csv_dir) 
    dice_df.to_csv(dice_csv_dir)
    return (all_volumes_data,dice_df) 







# colPairs_for_Dice = list(filter(  lambda tupl : (current_row[tupl[0]]!=" " and current_row[tupl[1]]!=" " ), for_dice_all_pairs)) 

# forDicePaths= list(map(lambda tupl: (current_row[tupl[0]],current_row[tupl[1]]),colPairs_for_Dice))


# files_df.columns


# studies = np.unique(files_df['study_id'].to_numpy())
# currentStudy = studies[0]
# locDf = files_df.loc[files_df['study_id'] == currentStudy]
# loc_series = np.unique(locDf['series_id'].to_numpy())


# series = np.unique(files_df['series_id'].to_numpy())
# currentSeries = series[31]
# locDf = files_df.loc[files_df['series_id'] == currentSeries]
# locDf

# currentSeries = loc_series[1]
# series_df = files_df.loc[files_df['series_id'] == currentSeries]

# series_doctors= np.unique(series_df['doctor_id'].to_numpy())


# locDf['series_id'].to_numpy()

# series_df.columns

