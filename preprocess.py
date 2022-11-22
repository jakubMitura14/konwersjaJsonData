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
from mainFuncs import getLabelsAbbrev


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

def get_indicies_to_zero(current_row,negatedProstate,colOfIntrA,colOfIntrB):
    """
    compares two volumes in case of overlap set the found entries to 0 and ovewrite the files
    """
    # print(f" colOfIntrA {colOfIntrA} colOfIntrB {colOfIntrB} ")
    pathA = current_row[colOfIntrA]
    pathB = current_row[colOfIntrB]

    imageA=sitk.ReadImage(pathA)
    imageB=sitk.ReadImage(pathB)

    dataA=sitk.GetArrayFromImage(imageA).astype(bool)
    dataB=sitk.GetArrayFromImage(imageB).astype(bool)
    #looking for overlap and for voxels present here but not in prostate

    return np.logical_or(np.logical_and(negatedProstate,dataB ) ,np.logical_and(negatedProstate,dataA ))


def get_common_indicies(current_row,negatedProstate,colOfIntrA,colOfIntrB):
    """
    compares two volumes in case of overlap set the found entries to 0 and ovewrite the files
    """
    pathA = current_row[colOfIntrA]
    pathB = current_row[colOfIntrB]

    imageA=sitk.ReadImage(pathA)
    imageB=sitk.ReadImage(pathB)

    dataA=sitk.GetArrayFromImage(imageA).astype(bool)
    dataB=sitk.GetArrayFromImage(imageB).astype(bool)
    #looking for overlap and for voxels present here but not in prostate
    common = np.logical_and(dataA,dataB)
    common_neg=np.logical_not(common)
    res= np.logical_and(common_neg, dataA )

    save_from_arr(res.astype(np.int16),imageA,pathA)



    return common



def get_bool_arr_from_path(colName,current_row):
    """
    given path reads it and return associated array
    then it casts it to boolean data type
    """
    path= current_row[colName]
    imageA=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(imageA).astype(bool)


def getModDistance(indexTop,indexIn):
    res=((abs(indexTop[0]-indexIn[0]))
        +(abs(indexTop[1]-indexIn[1]))
        +(abs(indexTop[2]-indexIn[2]))
        +(abs(indexTop[0]-indexIn[0])*3))  
    return res 

def sortAndGetSubsection(indList,indexTop,numbToAnalyze):
    res=list(map(lambda indexIn: getModDistance(indexTop,indexIn)  ,indList))
    if(len(res)>=numbToAnalyze):
        res=list(sorted(res))[0:numbToAnalyze]
        return np.mean(res)
    return 10000.0    



def getClosestIndex2D(indexTop,boolArrs_indicies):
    """
    given cartesian index it return the label number with some index closest to queried index
    """
    numbToAnalyze=5
    boolArrs_indiciesIn= list(map(lambda indList : sortAndGetSubsection(indList,indexTop,numbToAnalyze) ,boolArrs_indicies ))
    #print(np.min(boolArrs_indiciesIn))
    return np.where(boolArrs_indiciesIn == np.min(boolArrs_indiciesIn))[0][0]
   



def augment_indicies2D(indicies_to_mod,boolArrs_indicies):
    """
    go through all indicies that needs to be modified and associate index of the label with ith that should be set
    so it return original cartesian index plus index of the array where this index should be set to true
    """ 
    res= list(map( lambda index : (index[0],index[1],index[2] ,getClosestIndex2D(index, boolArrs_indicies)) ,indicies_to_mod))
    return res


def grow_labels(current_row,labelsOfIntrest,indicies_around,annot,prostateLab,indicies_around_full):
    """
    iterate over data - looks for any overlap between labels in labelsOfIntrest aand set such voxel to 0
    additionally it also take into account voxels that were not marked as any part of the prostate but are present in some subsection of 
    
    both groups will be then processed and will be changed into a label of the closest label
    """
    
    current_row=current_row[1]
    all_labels_types=np.unique(annot['labelName'].to_numpy())
    
    labelsOfIntrest_inner= list(filter(lambda labell: current_row[labell]!=" ",labelsOfIntrest )  )
    if(len(labelsOfIntrest_inner)>1):
        # getting powerset in order to simplify futher iterations
        print(f"labelsOfIntrest_inner {labelsOfIntrest_inner}")
        cart_prod=list(more_itertools.powerset(labelsOfIntrest_inner))
        cart_prod=list(filter(lambda tupl:len(tupl)==2  ,cart_prod))
        prostateBool = get_bool_arr_from_path(prostateLab,current_row )
        negatedProstate=np.logical_not(prostateBool)

        #resolves arbitrary overlaps between two masks
        def get_indicies_to_resolve(colOfIntrA,colOfIntrB):
            if(colOfIntrA==' ' or colOfIntrB==' '):
                return np.zeros(np.shape(prostateBool), dtype=bool  )    
            return get_common_indicies(current_row,negatedProstate,colOfIntrA,colOfIntrB)

        list(map( lambda tupl :  get_indicies_to_resolve(tupl[0],tupl[1]) ,cart_prod ))


        def get_indicies_to_zeroLoc(colOfIntrA,colOfIntrB):
            if(colOfIntrA==' ' or colOfIntrB==' '):
                return np.zeros(np.shape(prostateBool), dtype=bool  )    
            return get_indicies_to_zero(current_row,negatedProstate,colOfIntrA,colOfIntrB)



        #indicies to set To zero
        #itertools.accumulate(cart_prod, lambda x, y: x+y))
        # print(f"cart_prod {cart_prod} ")
        boolArrs = list(map( lambda colName :get_bool_arr_from_path(colName,current_row ) ,labelsOfIntrest_inner))
        
        
        if(len(cart_prod)>0):
            toSetToZeroBoolArrs= list(map( lambda tupl :  get_indicies_to_zeroLoc(tupl[0],tupl[1]) ,cart_prod ))
            toSetToZero=functools.reduce(np.logical_or, toSetToZeroBoolArrs)     
            negated_toSetToZero = np.logical_not(toSetToZero)
            boolArrs= list(map(lambda arr : np.logical_and(arr,negated_toSetToZero )  ,boolArrs))
      
        #now we get what is True in at least one of arrays
        # print(" labelsOfIntrest_innerrrrrrrrrr bbbb ")
        # print(labelsOfIntrest_inner)
        common = functools.reduce(np.logical_or, boolArrs)
        #negate
        common_not = np.logical_not(common)
        #below we will have all indicies in prostate but not in other labels, or those that was overlapping in two labels
        voxels_to_mod = np.logical_and(common_not, prostateBool)        
        #time to get indicies of the voxels to modify
        indicies_to_mod =  np.argwhere(voxels_to_mod)
        #getting indicies of current positions of labels
        boolArrs_indicies=list(map(np.argwhere,boolArrs ))
        #getting the labels to indicies_to_mod 
        augmentedIndicies2D = augment_indicies2D(indicies_to_mod,boolArrs_indicies)
        # print(f"indicies_to_mod {indicies_to_mod} \n")

        #modifying bool arrs according to augmented indicies
        # print(augmentedIndicies)


        for aug_index in augmentedIndicies2D:
            boolArrs[aug_index[3]][aug_index[0],aug_index[1],aug_index[2]]=True

        #now we have modified the arrays we need to overwrite it 
        for index, label in enumerate(labelsOfIntrest_inner):
            #print(label)
            pathA= current_row[label]
            image3D=sitk.ReadImage(pathA)
            pathB=  pathA#pathA.replace(".nii.gz", "_b_.nii.gz")
            #print(pathB)
            save_from_arr(boolArrs[index].astype(np.int16),image3D,pathB)



def dilatate_erode_conditionally(files_df,labelsOfIntrest,prostateLab ,annot):
    """
    main function that in parallel applies grow_labels
    """
    #usefull to iterate around not including iteration in z direction
    indicies_around=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([0])))
    indicies_around_full=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([-1,0,1])))

    # using only those rows where we have prostate
    prostateLab=f"{getLabelsAbbrev(prostateLab)}_noSeg"
    frame_of_intr=files_df.loc[files_df[prostateLab]!=" "]
    labelsOfIntrest=list(map( getLabelsAbbrev,labelsOfIntrest ))
    labelsOfIntrest=list(map( lambda el: f"{el}_noSeg",labelsOfIntrest ))

    # list(map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab,indicies_around_full=indicies_around_full), list(frame_of_intr.iterrows())))

    with mp.Pool(processes = mp.cpu_count()) as pool:
        pool.map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab,indicies_around_full=indicies_around_full), list(frame_of_intr.iterrows()))





# one can get the prostate without missing by getting all coordinates - group by key that x and y are equal set all coordinates to true if z is between z min and z max
