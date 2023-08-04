import SimpleITK as sitk
from subprocess import Popen
import subprocess
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import functools
from functools import partial
import sys
import os.path
from os import path as pathOs
import numpy as np
import tempfile
import shutil
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
import fileinput
import re
import subprocess
from toolz.itertoolz import groupby
import seaborn as sns
import einops
import matplotlib.pyplot as plt
import itertools
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
import itertools
from pydicom.fileset import FileSet
from os import path as pathOs
from pathlib import Path
import toolz
from toolz.curried import pipe, map, filter, get
from toolz import curry
from os.path import basename, dirname, exists, isdir, join, split
import nnunetv2

import elastixRegister as elastixRegister
from elastixRegister import reg_a_to_b
import tempfile
import shutil
import re
from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import os
from subprocess import Popen
import subprocess
from skimage.segmentation import mark_boundaries


def copy_changing_type(source, dest):
    image= sitk.ReadImage(source)
    # nan_count=np.sum(np.isnan(np.array(sitk.GetArrayFromImage(image)).flatten()))
    # if(nan_count>0):
    #     raise ValueError(f"!!! nan in {source}")
    image = sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    image=sitk.Cast(image, sitk.sitkFloat32)
    writer = sitk.ImageFileWriter() 
    writer.SetFileName(dest)
    writer.Execute(image)
    return dest

#metadata directory
resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
#directory with inferred prostates
dir_inferred_prost='/workspaces/konwersjaJsonData/explore/all_prost_segm_full_files/my_prost_infered'
sourceFrame = pd.read_csv(resCSVDir)
test_ids = pd.read_csv('/workspaces/konwersjaJsonData/explore/test_ids.csv' )
new_col_name= 'inferred_pg'




def get_id_from_file_name(path_str):
    path_str=path_str.replace('.nii.gz','')
    path_str=path_str[1:5]
    return int(path_str)

def add_t2w_to_name(source):
    if(source==' '):
        return ' '
    if('t2w' in source):
        return source
    new_path= source.replace('.nii.gz','_t2w.nii.gz')
    copy_changing_type(source, new_path)
    return new_path

def add_inferred_full_prost_to_dataframe(dir_inferred_prost, df,new_col_name):
    """ 
    we have some inferred anatomical segmentations done by previous 
    models now we want to take the folder with 
    """
    list_files= os.listdir(dir_inferred_prost)
    list_files= list(filter(lambda el : el[0]=='9' ,list_files ))
    list_ids= list(map(get_id_from_file_name,list_files))
    list_files= list(map( lambda el: f"{dir_inferred_prost}/{el}" ,list_files))
    file_and_id= dict(list(zip(list_ids,list_files)))
    new_col_dat= list(map( lambda el: file_and_id.get(el,' ') ,df['masterolds'].to_numpy() ))
    #changing path name to mark it is t2w related
    new_col_dat= list(map(add_t2w_to_name,new_col_dat))

    df[new_col_name]=new_col_dat
    return df



cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))

sourceFrame=add_inferred_full_prost_to_dataframe(dir_inferred_prost, sourceFrame,new_col_name)






def groupByMaster(rowws):
    grouped_by_master= groupby(lambda row : row[1]['masterolds'],rowws)
    # grouped_by_master=[(key,list(group)) for key, group in grouped_by_master]
    return dict(grouped_by_master).items()



def get_bool_arr_from_path(pathh):
    """
    given path reads it and return associated array
    then it casts it to boolean data type
    """
    imageA=sitk.ReadImage(pathh)
    return sitk.GetArrayFromImage(imageA).astype(bool)


def getPathsFromRow(row,list_columns):
    """
    extracting all paths of intrest from row
    """
    res=  map( lambda colName : (colName,row[1][colName] ),list_columns )
    return res

def getListModality(modalityName,pathhs,non_mri_inputs):
    """
    getting paths related to single modality and extracting main MRI image
    non_mri_inputs - some inputs that are designed to be put into input channels 
    """
    if(modalityName not in non_mri_inputs):
        # we are intrested only in paths that has the prostate segmentation
        pathhs=list(map(lambda el: el[1],pathhs))
        # pathhs= list(filter(lambda el :"pg_t2w.nii.gz" not in el , pathhs))
        mod_paths = list(filter(lambda pathh :modalityName in  pathh,pathhs))
        mri = list(filter(lambda el: '.mha' in el ,mod_paths))
        if(len(mri)==0):
            return ' ',[]
        mri=mri[0]   
        mod_paths= list(filter(lambda pathh: '.mha' not in pathh , mod_paths))
        return (modalityName,(mri,np.unique(mod_paths).tolist()))
    
    elif(modalityName in non_mri_inputs):
        # colNames=list(map(lambda el: el[0],pathhs))
        pathhss= list(filter(lambda el :modalityName in el[0] , pathhs))   
        if(len(pathhss)==0):
            return ' ',[]        
        res= (modalityName, (modalityName,np.unique(pathhss[0][1]).tolist())  )
        return res


def myFlatten(liist):
    return  itertools.chain(*liist)

def map_modalities(pathhs,modalities,non_mri_inputs):
    all_modalities=modalities+non_mri_inputs
    res= toolz.pipe(all_modalities
                ,map(partial(getListModality,pathhs=pathhs,non_mri_inputs=non_mri_inputs))
                ,list
            )
    # print(f"gggg {res}")
    return res



def iterGroupModalities(groupTuple,modalities_of_intrest,label_cols,non_mri_inputs ):
    """
    grouping the paths into dictionary relative to modalities they represent and lesions on thise 
    modalities
    """
    masterOlds, listRows= groupTuple
    pathhs=toolz.pipe(listRows
                ,map(partial(getPathsFromRow,list_columns=np.unique(label_cols+['series_MRI_path']+non_mri_inputs)))
                ,myFlatten
                # ,filter(lambda el : len(el)>2)
                ,list
                ,partial(map_modalities,modalities=modalities_of_intrest,non_mri_inputs=non_mri_inputs)
                ,dict
                )   
    return (masterOlds,pathhs)



def get_bool_or(pathA,pathB):
    if(isinstance(pathA, str) and isinstance(pathB, str)):
        return np.logical_or(get_bool_arr_from_path(pathA),get_bool_arr_from_path(pathB))
    elif(isinstance(pathA, str) and not isinstance(pathB, str)):
        return np.logical_or(get_bool_arr_from_path(pathA),pathB)
    elif(not isinstance(pathA, str) and isinstance(pathB, str)):
        return np.logical_or(get_bool_arr_from_path(pathB),pathA)
    else:
        return np.logical_or(pathB,pathA)

def get_4_id(masterolds):
    """
    take master id and changes it into string that starts with 0s and have always length 4
    """
    masteroldsStand=str(masterolds)
    if(len(masteroldsStand)==1):
        return f"000{masteroldsStand}"
    elif(len(masteroldsStand)==2):
        return f"00{masteroldsStand}"
    elif(len(masteroldsStand)==3):
        return f"0{masteroldsStand}"
    return masteroldsStand

def save_from_arr(zeroArray,image3D,newPathLab):
    """
    given array saves it to file into defined path using simpleitk
    """
    writer = sitk.ImageFileWriter()
    image = sitk.GetImageFromArray(zeroArray.astype(float).astype(np.uint8))  
    nan_count=np.sum(np.isnan(np.array(sitk.GetArrayFromImage(image)).flatten()))
    if(nan_count>0):
        raise ValueError(f"!!! nan in image would be saved as {newPathLab}")

    image.SetSpacing(image3D.GetSpacing())
    image.SetOrigin(image3D.GetOrigin())
    image.SetDirection(image3D.GetDirection())   
    image = sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    writer.SetFileName(newPathLab)
    writer.Execute(image)

def copy_changing_type(source, dest):
    image= sitk.ReadImage(source)
    # nan_count=np.sum(np.isnan(np.array(sitk.GetArrayFromImage(image)).flatten()))
    # if(nan_count>0):
    #     raise ValueError(f"!!! nan in {source}")
    image = sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    image=sitk.Cast(image, sitk.sitkFloat32)
    writer = sitk.ImageFileWriter() 
    writer.SetFileName(dest)
    writer.Execute(image)
    return dest

# mod="adc"
def get_key_by_value(mod,channel_names):
    return list(channel_names.keys())[list(channel_names.values()).index(mod)]

def prepare_out_paths(group,modalities_of_intrest,labelsTrFolder,imagesTrFolder,non_mri_inputs,channel_names ):
    #preparing names
    for_id=get_4_id(group[0])
    label_new_path= join(labelsTrFolder,f"9{for_id}00.nii.gz" )
    # prostate_path=join(imagesTrFolder,f"9{for_id}00_000{3}.nii.gz" )
    out_pathsDict= list(map( lambda mod:(mod,join(imagesTrFolder,f"9{for_id}00_000{get_key_by_value(mod,channel_names)}.nii.gz" )) ,np.unique(modalities_of_intrest+non_mri_inputs) ))
    out_pathsDict=dict(out_pathsDict)
    return label_new_path,out_pathsDict

# with mp.Pool(processes = mp.cpu_count()) as pool:
# #with mp.Pool(processes = 1) as pool:
#     @curry  
#     def pmap(fun,iterable):
#         return pool.map(fun,iterable)
    
def reg_a_to_b_by_metadata_single_b(fixed_image_path,moving_image_path):
    # print(f"fixed_image_path {fixed_image_path} moving_image_path {moving_image_path}")
    # moving_image_path=moving_image_path[0]
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)

    # fixed_image=sitk.Cast(fixed_image, sitk.sitkUInt8)
    # moving_image=sitk.Cast(moving_image, sitk.sitkInt)
    
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkBSpline, 0)
    return sitk.GetArrayFromImage(resampled)


def visualize_range(group,main_modality,modalities_of_intrest,non_mri_inputs):

    modalit_path_add= list(map( lambda el:(group[1][el]) ,non_mri_inputs))
    # print(f"ggg group[1] {group[1]}")
    # print(f"modalit_path_add[0][1] {modalit_path_add[0][1]}")
    if(modalit_path_add[0][1][0]==' '):
        print(f"no prostate! ")
        return " "

    modalities_of_intrest_without_main= list(filter( lambda el: el!=main_modality , modalities_of_intrest))   
    modalities_of_intrest_without_main=non_mri_inputs+modalities_of_intrest_without_main
    # print(f"modalities_of_intrest_without_main {modalities_of_intrest_without_main}")

    sources_dict=group[1]
    sources_dict[non_mri_inputs[0]]=(modalit_path_add[0][1][0],)


    registered_modalities_arrs= list(map(lambda mod: reg_a_to_b_by_metadata_single_b(sources_dict[main_modality][0],sources_dict[mod][0])                                    
                                                      ,modalities_of_intrest_without_main ))
    
    adc_arr=sitk.GetArrayFromImage(sitk.ReadImage(group[1][main_modality][0]))
    prostate_arr= registered_modalities_arrs[1]
    hbv_arr= registered_modalities_arrs[2]

    prostate_arr_indicies= np.argwhere(prostate_arr)

    max_z= np.max(prostate_arr_indicies[:,0])
    min_z= np.min(prostate_arr_indicies[:,0])

    max_x= np.max(prostate_arr_indicies[:,1])
    min_x= np.min(prostate_arr_indicies[:,1])

    max_y= np.max(prostate_arr_indicies[:,2])
    min_y= np.min(prostate_arr_indicies[:,2])

    add_padd=5

    adc_arr=adc_arr[]


    big_prost_z=einops.reduce(prostate_arr,'z x y-> z','sum')
    slice=np.argmax(big_prost_z)

    
    prost= prostate_arr[slice,:,:]
    image_to_disp_big= adc_arr[slice,:,:]

    with_boundaries=mark_boundaries(image_to_disp_big, prost.astype(int) )
    plt.imshow(with_boundaries)
    plt.show()

    # registered_modalities= list(map(lambda mod: reg_a_to_b(join(temp_dir,mod),group[0],group[1][main_modality][0],group[1][mod][0],group[1][mod][1],reg_prop
    #                                                             ,elacticPath,transformix_path,mod)
                        # ,modalities_of_intrest_without_main   ))

filter_ids=lambda row: row[1]['masterolds'] not in test_ids
modalities_of_intrest=['t2w','adc','hbv']
cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))
main_modality = 'adc'
non_mri_inputs=[new_col_name]

# with mp.Pool(processes = mp.cpu_count()) as pool:
# with mp.Pool(processes = 1) as pool:
#     @curry  
#     def pmap(fun,iterable):
#         return pool.map(fun,iterable)

toolz.pipe(sourceFrame.iterrows()
                                ,filter(lambda row: row[1]['series_desc'] in modalities_of_intrest)
                                ,filter(filter_ids) # filter out all of the test cases
                                ,groupByMaster
                                ,map(partial(iterGroupModalities,modalities_of_intrest=modalities_of_intrest,label_cols=lesion_cols,non_mri_inputs=non_mri_inputs))
                                ,filter(lambda group: ' ' not in group[1].keys() )
                                ,list
                                ,map(partial(visualize_range,main_modality=main_modality,modalities_of_intrest=modalities_of_intrest,non_mri_inputs=non_mri_inputs))                            
                                ,list)