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

# ### first one need to invoke the get3dFiles to produce dataframe and save it in resCSVDir
# resCSVDir='/workspaces/konwersjaJsonData/resCSV'

# JSON = '/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_dataset_D_gQm1nQ_2022-07-15-104055.json'
# results = mdai.common_utils.json_to_dataframe(JSON)
# #so we have dictionary of dataframes
# results.keys()#'annotations', 'studies', 'labels'
# annot=results['annotations']

# files_df = pd.read_csv(resCSVDir)



def save_from_arr(zeroArray,image3D,newPathLab):
    writer = sitk.ImageFileWriter()
    image = sitk.GetImageFromArray(zeroArray)  
    image.SetSpacing(image3D.GetSpacing())
    image.SetOrigin(image3D.GetOrigin())
    image.SetDirection(image3D.GetDirection())    

    writer.SetFileName(newPathLab)
    writer.Execute(image)

def mutate_if_not_in_prostate(labelName,label_bool_arr, negatedProstate,dataPrim ):
    """
    get the indicies that are present in a given label but not in prostate as a whole
    """
    if(labelName!='urethra'):
        indd=np.logical_and(negatedProstate,label_bool_arr )
        dataPrim[indd]=0


def get_common_indicies(current_row,negatedProstate,colOfIntrA,colOfIntrB):
    """
    compares two volumes in case of overlap set the found entries to 0 and ovewrite the files
    """
    pathA = current_row[colOfIntrA]
    pathB = current_row[colOfIntrB]

    imageA=sitk.ReadImage(pathA)
    imageB=sitk.ReadImage(pathB)

    dataAprim=sitk.GetArrayFromImage(imageA)
    dataBprim=sitk.GetArrayFromImage(imageB)

    dataA=dataAprim.astype(bool)
    dataB=dataBprim.astype(bool)

    #looking for overlap
    common = np.logical_and(dataA,dataB)
    #setting overlap to 0
    dataAprim[common]=0
    dataBprim[common]=0

    #setting also those voxels that are not present in prostate as a whole to 0 if label is not urethra
    mutate_if_not_in_prostate(colOfIntrA,dataA, negatedProstate,dataAprim )
    mutate_if_not_in_prostate(colOfIntrB,dataB, negatedProstate,dataBprim )


    #overwriting files with changed entries
    save_from_arr(dataAprim,imageA,pathA)
    save_from_arr(dataBprim,imageB,pathB)
    
    # indicies_common = torch.t(common.to_sparse().indices()).type(torch.int32).contiguous()
    # return indicies_common

def get_bool_arr_from_path(colName,current_row):
    path= current_row[colName]
    imageA=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(imageA).astype(bool)



def look_around_to_true(cart_index,boolArrs,index,indicies_around):
    """
    we look around given coordinates if we will find  some true we return currect index
    if nothing was found we return -1
    """
    for index_change in indicies_around:
        #get coordinate of neighbour
        newX=cart_index[0]+index_change[0]
        newY=cart_index[1]+index_change[1]
        newZ=cart_index[2]+index_change[2]
        #if in the analyzed label we will find neighbour as true we will set it to True 
        if(boolArrs[index][newX,newY,newZ]):
            return index
    return -1

def find_new_label(cart_index,boolArrs,labelsOfIntrest_inner,indicies_around):

    #lloking for index - index is marking the label in labelsOfIntrest
    proposedIndicies=list(map(lambda index:look_around_to_true(cart_index,boolArrs,index,indicies_around)   ,range(0,len(labelsOfIntrest_inner) )))
    chosen_index = np.max(proposedIndicies)
    return (cart_index[0],cart_index[1],cart_index[2],chosen_index  )  


def modify_bool_arrs(current_row,boolArrs ,indicies_to_mod ,labelsOfIntrest_inner,indicies_around):

    if(len(indicies_to_mod)>0):#condition needed to make recursion possible
        augmented_indicies = list(map(lambda cart_index: find_new_label(cart_index,boolArrs,labelsOfIntrest_inner,indicies_around),indicies_to_mod))
        indiciesToMod= list( filter( lambda tupl: tupl[3]>=0  ,augmented_indicies))

        indiciesToGoFuther= list( filter( lambda tupl: tupl[3]<0  ,augmented_indicies))

        #mutating the arrays according to augmented indicies
        for aug_index in indiciesToMod:
            boolArrs[aug_index[3]][aug_index[0],aug_index[1],aug_index[2]]=True
        
        #we call recursively function in order to dilatate all needed voxels
        modify_bool_arrs(current_row,boolArrs ,indiciesToGoFuther,labelsOfIntrest_inner,indicies_around)

def grow_labels(current_row,labelsOfIntrest,indicies_around,annot,prostateLab):
    current_row=current_row[1]
    all_labels_types=np.unique(annot['labelName'].to_numpy())
    
    labelsOfIntrest_inner= list(filter(lambda labell: current_row[labell]!=" ",labelsOfIntrest )  )
    if(len(labelsOfIntrest_inner)>0):
        # getting powerset in order to simplify futher iterations
        cart_prod=list(more_itertools.powerset(labelsOfIntrest_inner))
        cart_prod=list(filter(lambda tupl:len(tupl)==2  ,cart_prod))
        prostateBool = get_bool_arr_from_path(prostateLab,current_row )
        negatedProstate=np.logical_not(prostateBool)
        #setting entries that are overlapping ina any two labels to 0
        list(map( lambda tupl : get_common_indicies(current_row,negatedProstate,*tupl),cart_prod))
        #now we want to find the entries that are 1 in 'prostate' but zero in labelsOfIntrest_inner masks
        boolArrs = list(map( lambda colName :get_bool_arr_from_path(colName,current_row ) ,labelsOfIntrest_inner))
        #now we get what is True in at least one of arrays
        print(labelsOfIntrest_inner)
        common = functools.reduce(np.logical_or, boolArrs)
        #negate
        common_not = np.logical_not(common)
        #below we will have all indicies in prostate but not in other labels, or those that was overlapping in two labels
        voxels_to_mod = np.logical_and(common_not, prostateBool)
        #time to get indicies of the voxels to modify
        indicies_to_mod =  np.argwhere(voxels_to_mod) #torch.t(voxels_to_mod.coalesce().contiguous().to_sparse().indices()).contiguous().coalesce().type(torch.int32).numpy()
        #got indicies of labels the will be used to dilatate given masks
        #now we need to use augmented indicies and execute the function one more time if some augmented indicies returned -1
        modify_bool_arrs(current_row,boolArrs,indicies_to_mod ,labelsOfIntrest_inner,indicies_around)
        #now we have modified the arrays we need to overwrite it 
        for index, label in enumerate(labelsOfIntrest_inner):
            print(label)
            pathA= current_row[label]
            image3D=sitk.ReadImage(pathA)
            save_from_arr(boolArrs[index].astype(np.int16),image3D,pathA)



# #first filter out only the rows with prostate as only this is intresting for us at the moment 
# all_labels_types=np.unique(annot['labelName'].to_numpy())
# prostateLab = 'prostate'
# labelsOfIntrest = ['peripheral zone',  'transition zone','anterior fibromuscular stroma', 'central zone', 'urethra']

def dilatate_erode_conditionally(files_df,labelsOfIntrest,prostateLab ,annot):
    #usefull to iterate around
    indicies_around=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([-1,0,1])))
    # using only those rows where we have prostate
    frame_of_intr=files_df.loc[files_df[prostateLab]!=" "]
    #list(map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest), list(frame_of_intr.iterrows())))
    #modify all arrays in parallel
    with mp.Pool(processes = mp.cpu_count()) as pool:
        pool.map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab), list(frame_of_intr.iterrows()))




# arrr=np.array([ [  [False,False  ],[False,False  ] ],[  [False,True  ],[False,True  ] ]  ])
# arrr.shape
# np.argwhere(arrr)


# grow_labels(current_row)

# current_row = list(frame_of_intr.iterrows())[0]

