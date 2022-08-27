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

def save_from_arr(zeroArray,image3D,newPathLab):
    """
    given array saves it to file into defined path using simpleitk
    """
    writer = sitk.ImageFileWriter()
    image = sitk.GetImageFromArray(zeroArray)  
    image.SetSpacing(image3D.GetSpacing())
    image.SetOrigin(image3D.GetOrigin())
    image.SetDirection(image3D.GetDirection())    

    writer.SetFileName(newPathLab)
    writer.Execute(image)


def get_volume(path):
    """
    get volume of a binary mask that resides in path
    """
    image3D=sitk.ReadImage(path)
    data=sitk.GetArrayFromImage(image3D)
    spacing = image3D.GetSpacing()
    #we get total vlume by multiplying sum of all voxels with each voxel volume
    volume = np.sum(data)*(spacing[0]*spacing[1]*spacing[2])/1000
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
    """
    given data of single series return the volume of each label 
    in case given label is not present return -1
    """
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
    """
    utility function getting label for each annotator and combines it into single string
    """
    locLocDf = list(locDf.loc[locDf['doctor_id'] == doctor_id].iterrows())[0][1]
    #we filter all lesion names that happen to be present with this annotator doctor
    liist= list( filter(lambda clN: locLocDf[clN]!=" " ,col_names_for_dice))
    liist= list( map(lambda clN: clN+'_@_'+locLocDf[clN] ,liist))
    #we will add also a path to the lesions of intrest
    return list(map(lambda entry : doctor_id+'_@_'+entry   ,liist))



def get_dice_from_tupl(tupl, locDf):
    """
    given tuple holding data about label and annotator return the associated DICE score
    """
    changeA= tupl[0].split('_@_')
    changeB= tupl[1].split('_@_')
    
    doctorA= changeA[0]
    doctorB= changeB[0]

    labelA = changeA[1]
    labelB = changeB[1]

    pathA = list(locDf.loc[locDf['doctor_id'] == doctorA].iterrows())[0][1][labelA]
    pathB = list(locDf.loc[locDf['doctor_id'] == doctorB].iterrows())[0][1][labelB]
    return get_dice_between(pathA,pathB)

def splitLabelDataForFrame(tupl):
    """
    given tupl with data return list that is ready to be row in the output dataframe
    """
    return(tupl[0][0]#series id
    ,tupl[0][1].split('_@_')[0] #doctor a
    ,tupl[0][1].split('_@_')[1] #lesion a
    ,tupl[0][1].split('_@_')[2] #path to lesion a
    ,tupl[0][2].split('_@_')[0] #doctor b
    ,tupl[0][2].split('_@_')[1] #lesion b
    ,tupl[0][2].split('_@_')[2] #path to lesion b
    ,tupl[1]#dice
    ,tupl[2]  #mriPath    
    ) 

def get_dice_in_all_pairs(currentSeries,preprocessed_df,col_names_for_dice):
    """
    analyzing all possible combination o lesions descriped by diffrent annotators
    calculating dice score between them and returning nested tuple with series id and dice score for each pair
    """
    
    locDf = preprocessed_df.loc[preprocessed_df['series_id'] == currentSeries]
    doctors= np.unique(locDf['doctor_id'].to_numpy())
    mriPath= np.unique(locDf['series_MRI_path'].to_numpy())[0]

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
        labels_per_doctor_all_pairs= list(map(lambda tupl: (tupl[0],tupl[1],mriPath  ) ,labels_per_doctor_all_pairs))

        
        #prepared for returning
        labels_per_doctor_all_pairs=list(map(splitLabelDataForFrame   ,labels_per_doctor_all_pairs))
        return labels_per_doctor_all_pairs
    return [(" ", " ", " ", " ", " ", " ", " ", " "," ")]

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

    #so now we have the series of nested tuples entry 1 is series uid second entry is list of tuples where fitst entry marks what
    #lesion and which annotator did it
    # goal isto  create dataframe where column
        #1) series id 2) first annotator id 3) first lesion name 4) second annotator id 5) socond lesion name 6) dice score
    dice_df = pd.DataFrame(list_dice_score, columns =['SeriesId', 'doctor_a', 'lesion_a','path_lesion_a','doctor_b','lesion_b','path_lesion_b', 'dice','mriPath'])
    volumes_frame.to_csv(volumes_csv_dir) 
    dice_df.to_csv(dice_csv_dir)
    return (all_volumes_data,dice_df) 


def labelNamesForDoc(coc,locDf):
    """
    given dataframe it looks for all of the names of lesions associated with given human annotator
    """
    rows = list(locDf.iterrows())
    rows = list(map(lambda row:row[1],rows))
    rows = list(filter(lambda row:row['doctor_a']==doc or row['doctor_b']==doc,rows))
    rows = list(map(lambda row:[row['lesion_a'],row['lesion_b']],rows))
    return np.unique(list(itertools.chain(*rows)))


def getLenInDoc(tupl):
    """
    helper function to find doctor with biggest number of lesions described
    """
    return len(tupl[1])

def getmaxDiceInDoc(row):
    """
    helper function to find doctor with biggest number of lesions described
    """
    return row['dice']

def choosePath(row, docName):
    """
    choose the path associated with human annotator
    """
    if(row['doctor_a']==docName):
        return row['path_lesion_a']
    return row['path_lesion_b'] 

def chooseLesionName(row, docName):
    """
    choose the path associated with human annotator
    """
    if(row['doctor_a']==docName):
        return row['lesion_a']
    return row['lesion_b'] 


lesion_a

dice_df.columns

#we have list of SeriesId,doctor_a,lesion_a,path_lesion_a,doctor_b,lesion_b,path_lesion_b,dice
#fisrt we will filter rows where dice is not 0
preprocessed_df=dice_df.loc[dice_df['dice'] != " "]
preprocessed_df=preprocessed_df.loc[preprocessed_df['dice'] != 0]
#then we get all unique series id
series = np.unique(preprocessed_df['SeriesId'].to_numpy())



currentSeries=series[0] # aaa
##per unique series id
locDf = preprocessed_df.loc[preprocessed_df['SeriesId'] == currentSeries]

#create folder with this series id and copy there the MRI mha file and original lesion files
locFolderPath=join(rootFolder_lesion_analysis,currentSeries)
os.makedirs(locFolderPath,exist_ok=True)
#get some mri path it should point to the same mri in all cases for this series
mriPath=list(locDf['mriPath'].to_numpy())[0]
#then  unique lesion names and doctor names from lesion_a and lesion_b in this series n_lesions
lesion_names= np.unique(np.concatenate([locDf['lesion_a'].to_numpy(),locDf['lesion_b'].to_numpy()]))
doctor_names= np.unique(np.concatenate([locDf['doctor_a'].to_numpy(),locDf['doctor_b'].to_numpy()]))
#**copy files
aZipped=list(zip(locDf['lesion_a'].to_numpy(),locDf['doctor_a'].to_numpy(),locDf['path_lesion_a'].to_numpy()    ))
bZipped=list(zip(locDf['lesion_b'].to_numpy(),locDf['doctor_b'].to_numpy(),locDf['path_lesion_b'].to_numpy()    ))
allZipped= list(np.concatenate([aZipped,bZipped]))
#get only unique
allZipped=list(map(lambda tupl: tupl[0]+'___' +tupl[1]+'___'+tupl[2], allZipped  ))
allZipped=np.unique(allZipped )
allZipped=list(map( lambda fused : fused.split("___")  ,allZipped ))
#name of the copied file will point to the lesion name and annotator
list(map( lambda tupl: shutil.copyfile(tupl[2] ,join( locFolderPath ,tupl[0]+'_'+tupl[1]+'nii.gz')  ),allZipped))
shutil.copyfile(mriPath, join(locFolderPath, 'volume.mha'))

#we select the doctor with most lesion names present
doctor_lesions=list(map(partial(labelNamesForDoc,locDf=locDf) ,doctor_names))
zipped= list(zip(doctor_names, doctor_lesions))
#data about human annotator with max number of lesions described
docTuplMax=max(zipped, key=getLenInDoc)
#now we keep doc max as reference we iterate over his/her lesions and associate it with lesions of other annotator with highest dice score with this lesion
maxDocName=docTuplMax[0]
lesions_to_analyze=docTuplMax[1]
doctors_not_max=list(filter(lambda docName:docName!=maxDocName  ,doctor_names))
rows=list(locDf.iterrows())
rows=list(map(lambda row: row[1] ,rows))



current_lesion=lesions_to_analyze[0]# krowa
#we choose sinlge lesion and get rows only associated with it
rows=list(filter(lambda row: 
                        (row['lesion_a']==current_lesion and row['doctor_a']==maxDocName)
                        or 
                        (row['lesion_b']==current_lesion and row['doctor_b']==maxDocName ),rows ))
#now we will analyze non chosen doctors and from their data we will choose the image that is most simmilar to chosen label of max doctor
rows_not_max_doc=list(map(lambda doctorNotMaxName :list(filter(lambda row: 
                        (row['doctor_a']==doctorNotMaxName or row['doctor_b']==doctorNotMaxName)
                        ,rows )) ,doctors_not_max))
perDocRows=list(map(lambda doctorNotMaxName :list(filter(lambda row: 
                        (row['doctor_a']==doctorNotMaxName or row['doctor_b']==doctorNotMaxName)
                        ,rows )) ,doctors_not_max))

perDocRows=list(map(lambda docRows : max(docRows, key=getmaxDiceInDoc)   ,perDocRows))

zipped_perDocRows= list(zip(doctors_not_max,perDocRows ))

paths_to_fuse= list(map( lambda tupl: choosePath(tupl[1],tupl[0]) ,zipped_perDocRows))
images_toFuse= list(map( sitk.ReadImage ,paths_to_fuse))
images_toFuse= list(map( sitk.GetArrayFromImage ,images_toFuse))
images_toFuse= list(map(lambda imDat : imDat.astype(bool) ,images_toFuse))
fused = functools.reduce(np.logical_and, images_toFuse)


namee=_row["doctor_a"]+'_'+row["lesion_a"]+'_'+row["doctor_b"]+'_'+row["lesion_b"]+'nii.gz'
newPath=join(locFolderPath,namee)
save_from_arr(commonPart.astype(np.uint8),imageA,newPath)





# #we have list of SeriesId,doctor_a,lesion_a,path_lesion_a,doctor_b,lesion_b,path_lesion_b,dice
