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

def get_common_indicies(current_row,negatedProstate,colOfIntrA,colOfIntrB):
    """
    compares two volumes in case of overlap set the found entries to 0 and ovewrite the files
    """
    print(f" colOfIntrA {colOfIntrA} colOfIntrB {colOfIntrB} ")
    pathA = current_row[colOfIntrA]
    pathB = current_row[colOfIntrB]

    imageA=sitk.ReadImage(pathA)
    imageB=sitk.ReadImage(pathB)

    dataA=sitk.GetArrayFromImage(imageA).astype(bool)
    dataB=sitk.GetArrayFromImage(imageB).astype(bool)
    #looking for overlap and for voxels present here but not in prostate
    common = np.logical_and(dataA,dataB)
    common = np.logical_or(np.logical_and(negatedProstate,dataA ) ,common)
    common = np.logical_or(np.logical_and(negatedProstate,dataB ) ,common)

    return common


def get_bool_arr_from_path(colName,current_row):
    """
    given path reads it and return associated array
    then it casts it to boolean data type
    """
    path= current_row[colName]
    imageA=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(imageA).astype(bool)



def look_around_to_true(cart_index,boolArrs,index,indicies_around,indicies_around_full):
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

# def find_new_label(cart_index,boolArrs,labelsOfIntrest_inner,indicies_around,indicies_around_full,indexx):
#     """
#     given some position it looks in all directions if will find given label in neighberhood 
#     will mark this voxel as a member of neghbouring label
#     """
#     proposedIndicies=[]
#     if(indexx%8==0):
#         proposedIndicies=list(map(lambda index:look_around_to_true(cart_index,boolArrs,index,indicies_around_full,indicies_around_full)   ,range(0,len(labelsOfIntrest_inner) )))
#     else:
#         proposedIndicies=list(map(lambda index:look_around_to_true(cart_index,boolArrs,index,indicies_around,indicies_around_full)   ,range(0,len(labelsOfIntrest_inner) )))

#     #loking for index - index is marking the label in labelsOfIntrest
#     #proposedIndicies=list(map(lambda index:look_around_to_true(cart_index,boolArrs,index,indicies_around)   ,range(0,len(labelsOfIntrest_inner) )))
#     chosen_index = np.max(proposedIndicies)
#     return (cart_index[0],cart_index[1],cart_index[2],chosen_index  )  


# def modify_bool_arrs(current_row,boolArrs ,indicies_to_mod ,labelsOfIntrest_inner,indicies_around,indicies_around_full):
#     """
#     on the basis of the information from find_new_label
#     mutates the array holding data about the image
#     """
#     indicies_to_modIn=indicies_to_mod
#     indexx=0
#     while(len(indicies_to_modIn)>0):#condition needed to make recursion possible
#         indent=indexx+1
#         augmented_indicies = list(map(lambda cart_index: find_new_label(cart_index,boolArrs,labelsOfIntrest_inner,indicies_around,indicies_around_full ,indexx),indicies_to_mod))
#         indiciesToMod= list( filter( lambda tupl: tupl[3]>=0  ,augmented_indicies))

#         indiciesToGoFuther= list( filter( lambda tupl: tupl[3]<0  ,augmented_indicies))

#         #mutating the arrays according to augmented indicies
#         for aug_index in indiciesToMod:
#             boolArrs[aug_index[3]][aug_index[0],aug_index[1],aug_index[2]]=True
#         indicies_to_modIn=indiciesToGoFuther

#         #we call recursively function in order to dilatate all needed voxels
#         #modify_bool_arrs(current_row,boolArrs ,indiciesToGoFuther,labelsOfIntrest_inner,indicies_around)


def getClosestIndex(index,boolArrs_indicies):
    """
    given cartesian index it return the label number with some index closest to queried index
    """
    #getting minimal distance (with added penalty for diffrent z coordinate) 
    minDists = list(map(lambda boolArrTupl : 
                            #getting euclidean distances with added cost if the points are not in the same layer (have diffrent z coordinates)
                            #and return the minimum distance preserving the index of the array
                            np.min(list(map(lambda point : np.linalg.norm(point - index) +((point[2]-index[2])*5)**2   ,boolArrTupl))   ) 
                            ,boolArrs_indicies) )
    # print("minDists")
    # print(np.where(minDists == np.min(minDists))[0])
    # print("  ---  ")
    #looking for index of array holding element closest to index
    indiciess=np.where(minDists == np.min(minDists))[0]
    return indiciess[len(indiciess)-1]


def augment_indicies(indicies_to_mod,boolArrs_indicies):
    """
    go through all indicies that needs to be modified and associate index of the label with ith that should be set
    so it return original cartesian index plus index of the array where this index should be set to true
    """ 
    return list(map( lambda index : (index[0],index[1],index[2] ,getClosestIndex(index, boolArrs_indicies)) ,indicies_to_mod))


def grow_labels(current_row,labelsOfIntrest,indicies_around,annot,prostateLab,indicies_around_full):
    """
    iterate over data - looks for any overlap between labels in labelsOfIntrest aand set such voxel to 0
    additionally it also take into account voxels that were not marked as any part of the prostate but are present in some subsection of prostate
    both groups will be then processed and will be changed into a label of the closest label
    """
    
    current_row=current_row[1]
    all_labels_types=np.unique(annot['labelName'].to_numpy())
    
    labelsOfIntrest_inner= list(filter(lambda labell: current_row[labell]!=" ",labelsOfIntrest )  )
    if(len(labelsOfIntrest_inner)>0):
        # getting powerset in order to simplify futher iterations
        cart_prod=list(more_itertools.powerset(labelsOfIntrest_inner))
        cart_prod=list(filter(lambda tupl:len(tupl)==2  ,cart_prod))
        #the same but without urethra
        cart_prod_no_urethra= list(filter(lambda tupl: tupl[0]!='urethra' and tupl[1]!='urethra'  ,cart_prod))
        print(" cart_prod  ")
        print(cart_prod)

        prostateBool = get_bool_arr_from_path(prostateLab,current_row )
        negatedProstate=np.logical_not(prostateBool)
        def get_common_indiciesLoc(colOfIntrA,colOfIntrB):
            if(colOfIntrA==' ' or colOfIntrB==' '):
                return np.zeros(np.shape(prostateBool), dtype=bool  )    
            return get_common_indicies(current_row,negatedProstate,colOfIntrA,colOfIntrB)



        #indicies to set To zero
        #itertools.accumulate(cart_prod, lambda x, y: x+y))
        
        toSetToZeroBoolArrs= list(map( lambda tupl :  get_common_indiciesLoc(tupl[0],tupl[1]) ,cart_prod_no_urethra ))
        toSetToZero=functools.reduce(np.logical_or, toSetToZeroBoolArrs)
        
        negated_toSetToZero = np.logical_not(toSetToZero)

        boolArrs = list(map( lambda colName :get_bool_arr_from_path(colName,current_row ) ,labelsOfIntrest_inner))
        boolArrs= list(map(lambda arr : np.logical_and(arr,negated_toSetToZero )  ,boolArrs))
      
        #now we get what is True in at least one of arrays
        print(" labelsOfIntrest_innerrrrrrrrrr  ")
        print(labelsOfIntrest_inner)
        common = functools.reduce(np.logical_or, boolArrs)
        #negate
        common_not = np.logical_not(common)
        #below we will have all indicies in prostate but not in other labels, or those that was overlapping in two labels
        voxels_to_mod = np.logical_and(common_not, prostateBool)        
        #time to get indicies of the voxels to modify
        indicies_to_mod =  np.argwhere(voxels_to_mod) #torch.t(voxels_to_mod.coalesce().contiguous().to_sparse().indices()).contiguous().coalesce().type(torch.int32).numpy()
        #getting indicies of current positions of labels
        boolArrs_indicies=list(map(np.argwhere,boolArrs ))
        # #adding indicies to associated boolean array
        # boolArrs_indicies=enumerate(boolArrs_indicies)
        #getting the labels to indicies_to_mod 
        augmentedIndicies = augment_indicies(indicies_to_mod,boolArrs_indicies)
        #modifying bool arrs according to augmented indicies
        # print(augmentedIndicies)
        for aug_index in augmentedIndicies:
            boolArrs[aug_index[3]][aug_index[0],aug_index[1],aug_index[2]]=True

        #now we have modified the arrays we need to overwrite it 
        for index, label in enumerate(labelsOfIntrest_inner):
            print(label)
            pathA= current_row[label]
            image3D=sitk.ReadImage(pathA)
            save_from_arr(boolArrs[index].astype(np.int16),image3D,pathA)



def dilatate_erode_conditionally(files_df,labelsOfIntrest,prostateLab ,annot):
    """
    main function that in parallel applies grow_labels
    """
    #usefull to iterate around not including iteration in z direction
    indicies_around=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([0])))
    indicies_around_full=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([-1,0,1])))

    # using only those rows where we have prostate
    frame_of_intr=files_df.loc[files_df[prostateLab]!=" "]
    #list(map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest), list(frame_of_intr.iterrows())))
    #modify all arrays in parallel
    #list(map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab,indicies_around_full=indicies_around_full), list(frame_of_intr.iterrows())))

    with mp.Pool(processes = mp.cpu_count()) as pool:
        pool.map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab,indicies_around_full=indicies_around_full), list(frame_of_intr.iterrows()))


# import SimpleITK as sitk
# import mdai
# import pandas as pd
# import numpy as np
# import cv2
# import pydicom
# import os
# import multiprocessing as mp
# import functools
# from functools import partial
# import mdai
# import math
# import time
# import mainFuncs
# import itertools
# import torch
# from os import path as pathOs
# import more_itertools

# # ### first one need to invoke the get3dFiles to produce dataframe and save it in resCSVDir
# # resCSVDir='/workspaces/konwersjaJsonData/resCSV'

# # JSON = '/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_dataset_D_gQm1nQ_2022-07-15-104055.json'
# # results = mdai.common_utils.json_to_dataframe(JSON)
# # #so we have dictionary of dataframes
# # results.keys()#'annotations', 'studies', 'labels'
# # annot=results['annotations']

# # files_df = pd.read_csv(resCSVDir)



# def save_from_arr(zeroArray,image3D,newPathLab):
#     writer = sitk.ImageFileWriter()
#     image = sitk.GetImageFromArray(zeroArray)  
#     image.SetSpacing(image3D.GetSpacing())
#     image.SetOrigin(image3D.GetOrigin())
#     image.SetDirection(image3D.GetDirection())    

#     writer.SetFileName(newPathLab)
#     writer.Execute(image)

# def mutate_if_not_in_prostate(labelName,label_bool_arr, negatedProstate,dataPrim ):
#     """
#     get the indicies that are present in a given label but not in prostate as a whole
#     """
#     if(labelName!='urethra'):
#         indd=np.logical_and(negatedProstate,label_bool_arr )
#         dataPrim[indd]=0


# def get_common_indicies(current_row,negatedProstate,colOfIntrA,colOfIntrB):
#     """
#     compares two volumes in case of overlap set the found entries to 0 and ovewrite the files
#     """
#     pathA = current_row[colOfIntrA]
#     pathB = current_row[colOfIntrB]

#     imageA=sitk.ReadImage(pathA)
#     imageB=sitk.ReadImage(pathB)

#     dataAprim=sitk.GetArrayFromImage(imageA)
#     dataBprim=sitk.GetArrayFromImage(imageB)

#     dataA=dataAprim.astype(bool)
#     dataB=dataBprim.astype(bool)

#     #looking for overlap
#     common = np.logical_and(dataA,dataB)
#     #setting overlap to 0
#     dataAprim[common]=0
#     dataBprim[common]=0

#     #setting also those voxels that are not present in prostate as a whole to 0 if label is not urethra
#     mutate_if_not_in_prostate(colOfIntrA,dataA, negatedProstate,dataAprim )
#     mutate_if_not_in_prostate(colOfIntrB,dataB, negatedProstate,dataBprim )


#     #overwriting files with changed entries
#     save_from_arr(dataAprim,imageA,pathA)
#     save_from_arr(dataBprim,imageB,pathB)
    
#     # indicies_common = torch.t(common.to_sparse().indices()).type(torch.int32).contiguous()
#     # return indicies_common

# def get_bool_arr_from_path(colName,current_row):
#     path= current_row[colName]
#     imageA=sitk.ReadImage(path)
#     return sitk.GetArrayFromImage(imageA).astype(bool)



# def look_around_to_true(cart_index,boolArrs,index,indicies_around,indicies_around_full):
#     """
#     we look around given coordinates if we will find  some true we return currect index
#     if nothing was found we return -1
#     """
#     for index_change in indicies_around:
#         #get coordinate of neighbour
#         newX=cart_index[0]+index_change[0]
#         newY=cart_index[1]+index_change[1]
#         newZ=cart_index[2]+index_change[2]
#         #if in the analyzed label we will find neighbour as true we will set it to True 
#         if(boolArrs[index][newX,newY,newZ]):
#             return index
#     return -1

# def find_new_label(cart_index,boolArrs,labelsOfIntrest_inner,indicies_around,indicies_around_full,indexx):

#     #lloking for index - index is marking the label in labelsOfIntrest we will look only in plane so change x and y if the indexx is not divisible by 30 in those rare cases when it is we will look in all directions
#     proposedIndicies=list(map(lambda index:look_around_to_true(cart_index,boolArrs,index,indicies_around_full,indicies_around_full)   ,range(0,len(labelsOfIntrest_inner) )))

#     # proposedIndicies=[]
#     # if(indexx%2==0):
#     #     proposedIndicies=list(map(lambda index:look_around_to_true(cart_index,boolArrs,index,indicies_around_full,indicies_around_full)   ,range(0,len(labelsOfIntrest_inner) )))
#     # else:
#     #     proposedIndicies=list(map(lambda index:look_around_to_true(cart_index,boolArrs,index,indicies_around,indicies_around_full)   ,range(0,len(labelsOfIntrest_inner) )))
#     chosen_index = np.max(proposedIndicies)
#     return (cart_index[0],cart_index[1],cart_index[2],chosen_index  )  


# def modify_bool_arrs(current_row,boolArrs ,indicies_to_mod ,labelsOfIntrest_inner,indicies_around,indicies_around_full):
#     indicies_to_modIn=indicies_to_mod
#     indexx=0
#     while(len(indicies_to_modIn)>0):
#         indexx=indexx+1
#         augmented_indicies = list(map(lambda cart_index: find_new_label(cart_index,boolArrs,labelsOfIntrest_inner,indicies_around,indicies_around_full,indexx),indicies_to_mod))
#         indiciesToMod= list( filter( lambda tupl: tupl[3]>=0  ,augmented_indicies))

#         indiciesToGoFuther= list( filter( lambda tupl: tupl[3]<0  ,augmented_indicies))
#         #print(f"indiciesToGoFuther  {len(indiciesToGoFuther)} {indicies_around}")

#         #mutating the arrays according to augmented indicies
#         for aug_index in indiciesToMod:
#             boolArrs[aug_index[3]][aug_index[0],aug_index[1],aug_index[2]]=True
        
#         indicies_to_modIn=indiciesToGoFuther

#         #we call recursively function in order to dilatate all needed voxels
#         # modify_bool_arrs(current_row,boolArrs ,indiciesToGoFuther,labelsOfIntrest_inner,indicies_around)

# def grow_labels(current_row,labelsOfIntrest,indicies_around,annot,prostateLab,indicies_around_full):
#     current_row=current_row[1]
#     all_labels_types=np.unique(annot['labelName'].to_numpy())
    
#     labelsOfIntrest_inner= list(filter(lambda labell: current_row[labell]!=" ",labelsOfIntrest )  )
#     if(len(labelsOfIntrest_inner)>0):
#         # getting powerset in order to simplify futher iterations
#         cart_prod=list(more_itertools.powerset(labelsOfIntrest_inner))
#         cart_prod=list(filter(lambda tupl:len(tupl)==2  ,cart_prod))
#         prostateBool = get_bool_arr_from_path(prostateLab,current_row )
#         negatedProstate=np.logical_not(prostateBool)
#         #setting entries that are overlapping ina any two labels to 0
#         list(map( lambda tupl : get_common_indicies(current_row,negatedProstate,*tupl),cart_prod))
#         #now we want to find the entries that are 1 in 'prostate' but zero in labelsOfIntrest_inner masks
#         boolArrs = list(map( lambda colName :get_bool_arr_from_path(colName,current_row ) ,labelsOfIntrest_inner))
#         #now we get what is True in at least one of arrays
#         common = functools.reduce(np.logical_or, boolArrs)
#         #negate
#         common_not = np.logical_not(common)
#         #below we will have all indicies in prostate but not in other labels, or those that was overlapping in two labels
#         voxels_to_mod = np.logical_and(common_not, prostateBool)
#         #time to get indicies of the voxels to modify
#         indicies_to_mod =  np.argwhere(voxels_to_mod) #torch.t(voxels_to_mod.coalesce().contiguous().to_sparse().indices()).contiguous().coalesce().type(torch.int32).numpy()
#         #got indicies of labels the will be used to dilatate given masks
#         #now we need to use augmented indicies and execute the function one more time if some augmented indicies returned -1
#         modify_bool_arrs(current_row,boolArrs,indicies_to_mod ,labelsOfIntrest_inner,indicies_around,indicies_around_full)
#         #now we have modified the arrays we need to overwrite it 
#         for index, label in enumerate(labelsOfIntrest_inner):
#             pathA= current_row[label]
#             image3D=sitk.ReadImage(pathA)
#             save_from_arr(boolArrs[index].astype(np.int16),image3D,pathA)



# # #first filter out only the rows with prostate as only this is intresting for us at the moment 
# # all_labels_types=np.unique(annot['labelName'].to_numpy())
# # prostateLab = 'prostate'
# # labelsOfIntrest = ['peripheral zone',  'transition zone','anterior fibromuscular stroma', 'central zone', 'urethra']

# def dilatate_erode_conditionally(files_df,labelsOfIntrest,prostateLab ,annot):
#     #usefull to iterate around
#     indicies_around=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([0])))
#     indicies_around_full=list(itertools.product(set([-1,0,1]),set([-1,0,1]),set([-1,0,1])))
#     # using only those rows where we have prostate
#     frame_of_intr=files_df.loc[files_df[prostateLab]!=" "]
#     #list(map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest), list(frame_of_intr.iterrows())))
#     #modify all arrays in parallel
#     list(map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab,indicies_around_full=indicies_around_full), list(frame_of_intr.iterrows())))


#     # with mp.Pool(processes = mp.cpu_count()) as pool:
#     #     pool.map(partial(grow_labels,labelsOfIntrest=labelsOfIntrest,indicies_around=indicies_around,annot=annot,prostateLab=prostateLab), list(frame_of_intr.iterrows()))




# # arrr=np.array([ [  [False,False  ],[False,False  ] ],[  [False,True  ],[False,True  ] ]  ])
# # arrr.shape
# # np.argwhere(arrr)


# # grow_labels(current_row)

# # current_row = list(frame_of_intr.iterrows())[0]



