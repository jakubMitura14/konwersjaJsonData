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

import nnunet.elastixRegister as elastixRegister
from nnunet.elastixRegister import reg_a_to_b,reg_a_to_b_itk
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

elacticPath='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/elastix'
transformix_path='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/transformix'
reg_prop='/workspaces/konwersjaJsonData/registration/parameters.txt'  


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





def myFlatten(liist):
    return  itertools.chain(*liist)


def getPathsFromRows(colName,listRows):
    """
    extracting all paths of intrest from row
    """
    res=  list(map( lambda row : row[1][colName] ,listRows ))
    return list(filter(lambda el: el!=' ',res ))


def iterGroups(groupTuple,adc_lesion_cols,anatomy_cols ):
    masterOlds, listRows= groupTuple
    #first find all of the lesion paths and anatomy paths
    anatomy_paths=toolz.pipe(anatomy_cols
                ,map(partial(getPathsFromRows,listRows=listRows))
                ,list) 
    lesion_paths=toolz.pipe(adc_lesion_cols
                ,map(partial(getPathsFromRows,listRows=listRows))
                ,list) 
    anatomy_paths=list(filter(lambda  el: len(el)>0 , anatomy_paths))
    lesion_paths=list(filter(lambda  el: len(el)>0 , lesion_paths))

    mri_paths = list(map(lambda row: (row[1]['series_desc'],row[1]['series_MRI_path']) ,listRows))
    adc = list(filter(lambda el: el[0]== 'adc' ,mri_paths ))[0]
    t2w = list(filter(lambda el: el[0]== 't2w' ,mri_paths ))[0]
    # now we need a path to t2w and adc
    return (masterOlds,adc[1],t2w[1],list(zip(anatomy_cols,anatomy_paths)),list(zip(adc_lesion_cols,lesion_paths )))

def remove_lesions_from_anatomy(anatomy_bool,lesion_bools ):
    """ 
    removes all areas of lesions from each anatomic area
    """
    reduced = np.array(functools.reduce(np.logical_or, lesion_bools))
    return np.logical_and(anatomy_bool,np.logical_not(reduced))

def get_mean_adc(bool_indexes,adc_path):
    if(len(bool_indexes.shape)==4):
        bool_indexes=bool_indexes[0,:,:,:]
    image_adc=sitk.ReadImage(adc_path)
    adc_data=sitk.GetArrayFromImage(image_adc)
    chosen_adc=adc_data[bool_indexes]
    return np.mean(chosen_adc.flatten())

def get_volume(path,bool_volume):
    """
    get volume of a binary mask 
    """
    image3D=sitk.ReadImage(path)
    # data=sitk.GetArrayFromImage(image3D)
    spacing = image3D.GetSpacing()
    #we get total vlume by multiplying sum of all voxels with each voxel volume
    volume = np.sum(bool_volume)*(spacing[0]*spacing[1]*spacing[2])/1000
    return volume


def register_in_group(groupTuple,temp_dir):
    masterOlds,adc,t2w,anatomy_paths,lesion_paths=groupTuple
    anatomy_names,anatomy_paths_pure = list(toolz.sandbox.core.unzip( anatomy_paths ))
    
    anatomy_names=list(anatomy_names)
    anatomy_paths_pure=list(anatomy_paths_pure)

    #registering t2w on adc
    loc_temp_dir=f"{temp_dir}/{masterOlds}"
    # loc_temp_dir=f"/workspaces/konwersjaJsonData/explore/debug/{masterOlds}" # TODO remove unhash above krowa

    os.makedirs(loc_temp_dir ,exist_ok = True)
    regg=reg_a_to_b(loc_temp_dir,masterOlds,adc,t2w,anatomy_paths_pure,reg_prop ,elacticPath,transformix_path,'adc',reIndex=0)
    if(regg==' '):
        return ' '
    _,registered_t2w,registered_anatomies = regg
    #get the original lesion changes and registered anatomies to booleans
    anatomy_bools= list(map( get_bool_arr_from_path ,registered_anatomies ))

    mean_adcs=[]
    if(len(lesion_paths)>0):
        lesion_names,lesion_paths_pure = list(toolz.sandbox.core.unzip( lesion_paths ))
        lesion_names =list(lesion_names)
        lesion_paths_pure=list(lesion_paths_pure)
        lesion_bools= list(map( get_bool_arr_from_path ,lesion_paths_pure ))
        #removing lesions from anatomy regions
        anatomy_bools = list(map(partial(remove_lesions_from_anatomy,lesion_bools=lesion_bools),anatomy_bools))    
        mean_adcs= list(map(partial(get_mean_adc, adc_path=adc),anatomy_bools))
    else:
        mean_adcs= list(map(partial(get_mean_adc, adc_path=adc) ,anatomy_bools))

    anatomy_bools_vols= list(itertools.starmap( get_volume ,zip(registered_anatomies,anatomy_bools)))




    return (masterOlds,dict(list(zip(anatomy_names, mean_adcs))),dict(list(zip(anatomy_names, anatomy_bools_vols)))  )
    

""" 
1) group by master id
2) register t2w to adc with elastix and all anatomic data using transfromix also on adc
3) iterate anatomic region by region
    a) subtract lesions from anatomic regions
    b) get anatomic region (what is left) to bool
    c) selest adc voxels using b
    d) get mean of selected voxels from c
"""

def save_mean_anatomy_adc(sourceFrame,anatomy_cols,anatomy_adc_csv_dir):

    cols=sourceFrame.columns
    cols=list(filter(lambda el: 'lesion' in el ,cols))
    adc_lesion_cols=list(filter(lambda el: 'adc_noSeg' in el ,cols))
    temp_dir = tempfile.mkdtemp()
    adc_means=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
    # with mp.Pool(processes = 1) as pool:
        @curry  
        def pmap(fun,iterable):
            return pool.map(fun,iterable)


        adc_means= toolz.pipe(sourceFrame.iterrows()
                                ,groupByMaster
                                ,pmap(partial(iterGroups,adc_lesion_cols=adc_lesion_cols,anatomy_cols=anatomy_cols ))
                                # ,filter(lambda group: ' ' not in group[1].keys() )
                                ,list
                                ,pmap(partial(register_in_group,temp_dir=temp_dir ))
                                ,list   
                                ,filter(lambda el: el!=' ') 
                                ,list                                                        )

    means_frame= pd.DataFrame()
    #populating frame with data
    means_frame['id']=list(map(lambda el : el[0]  ,adc_means))
    for col_name in anatomy_cols:
        means_frame[col_name]=list(map(lambda el : el[1].get(col_name,' ')  ,adc_means))
        means_frame[f"{col_name}_volume"]=list(map(lambda el : el[2].get(col_name,' ')  ,adc_means))

    means_frame.to_csv(anatomy_adc_csv_dir) 
    shutil.rmtree(temp_dir, ignore_errors=True)  

    return means_frame

# (masterOlds,dict(list(zip(anatomy_names, mean_adcs))))