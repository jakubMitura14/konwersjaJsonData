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
import nnunetv2

import elastixRegister
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

resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
elacticPath='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/elastix'
transformix_path='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/transformix'
reg_prop='/workspaces/konwersjaJsonData/registration/parameters.txt'  





def groupByMaster(rowws):
    grouped_by_master= groupby(lambda row : row[1]['masterolds'],rowws)
    # grouped_by_master=[(key,list(group)) for key, group in grouped_by_master]
    return dict(grouped_by_master).items()







sourceFrame = pd.read_csv(resCSVDir) 
cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))


main_modality = 't2w'
prostate_col= 'pg_noSeg' # name of the column with 

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
    res=  map( lambda colName :row[1][colName] ,list_columns )
    return res

def getListModality(modalityName,pathhs):
    """
    getting paths related to single modality and extracting main MRI image
    """
    if(modalityName!="prostate"):
        # we are intrested only in paths that has the prostate segmentation
        pathhs= list(filter(lambda el :"pg_t2w.nii.gz" not in el , pathhs))
        mod_paths = list(filter(lambda pathh :modalityName in  pathh,pathhs))
        mri = list(filter(lambda el: '.mha' in el ,mod_paths))
        if(len(mri)==0):
            return ' ',[]
        mri=mri[0]   
        mod_paths= list(filter(lambda pathh: '.mha' not in pathh , mod_paths))

        return (modalityName,(mri,mod_paths))
    elif(modalityName=="prostate"):
        pathhs= list(filter(lambda el :"pg_t2w.nii.gz" in el , pathhs))
        return (modalityName,(pathhs[0]))

def myFlatten(liist):
    return  itertools.chain(*liist)
def map_modalities(pathhs,modalities):
    return toolz.pipe(modalities
                ,map(partial(getListModality,pathhs=pathhs))
                ,list
            )



def iterGroupModalities(groupTuple,modalities_of_intrest,label_cols ):
    """
    grouping the paths into dictionary relative to modalities they represent and lesions on thise 
    modalities
    """
    masterOlds, listRows= groupTuple
    pathhs=toolz.pipe(listRows
                ,map(partial(getPathsFromRow,list_columns=label_cols+['series_MRI_path',prostate_col]))
                ,myFlatten
                ,filter(lambda el : len(el)>2)
                ,list
                ,partial(map_modalities,modalities=modalities_of_intrest+["prostate"])
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
    image = sitk.GetImageFromArray(zeroArray.astype(np.uint8))  
    image.SetSpacing(image3D.GetSpacing())
    image.SetOrigin(image3D.GetOrigin())
    image.SetDirection(image3D.GetDirection())    

    writer.SetFileName(newPathLab)
    writer.Execute(image)

def copy_changing_type(source, dest):
    image= sitk.ReadImage(source)
    writer = sitk.ImageFileWriter() 
    writer.SetFileName(dest)
    writer.Execute(image)
    return dest

def prepare_out_paths(group,modalities_of_intrest,labelsTrFolder,imagesTrFolder ):
    #preparing names
    for_id=get_4_id(group[0])
    label_new_path= join(labelsTrFolder,f"9{for_id}00.nii.gz" )
    prostate_path=join(imagesTrFolder,f"9{for_id}00_000{3}.nii.gz" )
    out_pathsDict= list(map( lambda mod:(mod,join(imagesTrFolder,f"9{for_id}00_000{modalities_of_intrest.index(mod)}.nii.gz" )) ,modalities_of_intrest))
    out_pathsDict=dict(out_pathsDict)
    return label_new_path,prostate_path,out_pathsDict

def process_labels_prim(labels,group,main_modality,label_new_path):
    # we get the sum of all labels 
    # reduced = np.array(toolz.sandbox.parallel.fold(get_bool_or, labels,map=map))
    reduced = np.array(functools.reduce(get_bool_or, labels))
    # now we need to save the sumed label and all of the MRIs 
    # we want to make it compatible with both nnunet in general and with the picai dataset so we will keep picai convention of numering cases
    # 0 t2w, 1 adc 2 hbv additionally we will set prostate gland label as 3 which will be output of the segmentation algorithm passed as preprocessing step
    # in order to avoid problems with repeating ids all ids from 9
    # we need also to add related labels        
    save_from_arr(reduced,sitk.ReadImage(group[1][main_modality][0]),label_new_path)


def add_files(group,main_modality,modalities_of_intrest,reg_prop,elacticPath,transformix_path,labelsTrFolder,imagesTrFolder,process_labels ):
    """
    first register images and their respective labels to t2w
    then reduces all labels into their sum
    then saves mri and reduced labels into nnunet workdir to get structure the same as in baseline picai nnunet algorithm
    """
    label_new_path,out_prostate_path,out_pathsDict=prepare_out_paths(group,modalities_of_intrest,labelsTrFolder,imagesTrFolder )
    #In case file already exist
    if(exists(out_pathsDict[main_modality])):
        out_pathsDict['label']=label_new_path 
        out_pathsDict['prostate']=out_prostate_path 
        return (group[0],out_pathsDict)


    temp_dir = tempfile.mkdtemp() # temporary directory
    modalities_of_intrest_without_main= list(filter( lambda el: el!=main_modality , modalities_of_intrest))
    #register all modalities and associated labels to main_modality
    registered_modalities= list(map(lambda mod: reg_a_to_b(join(temp_dir,mod),group[0],group[1][main_modality][0],group[1][mod][0],group[1][mod][1],reg_prop
                                                            ,elacticPath,transformix_path,mod)
                    ,modalities_of_intrest_without_main   ))
    # now we unzip to get 0) list of modalities 1) list of paths to main mris 2) list of lists of labels paths
    modalities,mris,labels=list(toolz.sandbox.core.unzip(registered_modalities))
    labels=list(toolz.concat(labels))
    modalities=list(modalities)
    modalities.append(main_modality)
    mris=list(mris)
    mris.append(group[1][main_modality][0])    

    #adding to the list the labels from main modality thay did not needed to be registered
    labels=labels+group[1][main_modality][1]

    if(len(labels)>0):

        process_labels(labels,group,main_modality,label_new_path)



    #zipping for starmap use
    zipped_modalit_path = list(zip(modalities,mris))
    zipped_modalit_path= list(map( lambda tupl:(tupl[1], out_pathsDict[tupl[0]]) ,zipped_modalit_path))
    
    #as we already have prepared the destination paths and sources for images we need now to copy files
    # we need to remember that we are  getting from mha to nii gz
    list(itertools.starmap(copy_changing_type ,zipped_modalit_path ))

    _,new_mri_paths= list(toolz.sandbox.core.unzip(zipped_modalit_path))
    new_mri_paths=list(new_mri_paths)
    newPaths= list(zip(modalities,new_mri_paths))
    newPaths.append(('label',label_new_path ))
    #copying label holding segmentation of full prostate gland
    currProstPath= group[1]["prostate"]
    shutil.copyfile(currProstPath,out_prostate_path )
    newPaths.append(('prostate',out_prostate_path))

    #clearing temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    return (group[0],dict(newPaths))


def main_prepare_nnunet(dataset_id, modalities_of_intrest,channel_names,label_names,label_cols,process_labels):
    """
    main function for preparing nnunet
    """


    #first removing old data
    nNunetBaseFolder='/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder'

    shutil.rmtree(nNunetBaseFolder)
    taskName= f"Dataset{dataset_id}_Prostate"
    taskFolder = join(nNunetBaseFolder,'nnUNet_raw',taskName)
    preprocesss_folder= join(nNunetBaseFolder,'nnUNet_preprocessed')
    results_folder= join(nNunetBaseFolder,'nnUNet_results')
    mainResults_folder="/home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results"

    imagesTrFolder= join(taskFolder,'imagesTr')
    labelsTrFolder= join(taskFolder,'labelsTr')
    imagesTsFolder= join(taskFolder,'imagesTs')
    json_path= join(taskFolder,'dataset.json')

    # main modality that will be set as a reference and others will be registered to it 


    os.makedirs(nNunetBaseFolder ,exist_ok = True)

    # os.makedirs(join(nNunetBaseFolder,'nnUNet_raw_data_base') ,exist_ok = True)
    # os.makedirs(join(nNunetBaseFolder,'nnUNet_raw_data_base','nnUNet_raw_data') ,exist_ok = True)
    os.makedirs(taskFolder ,exist_ok = True)
    os.makedirs(imagesTrFolder ,exist_ok = True)
    os.makedirs(labelsTrFolder ,exist_ok = True)
    os.makedirs(preprocesss_folder ,exist_ok = True)
    os.makedirs(results_folder ,exist_ok = True)
    os.makedirs(mainResults_folder ,exist_ok = True)
    os.makedirs(join(mainResults_folder,taskName),exist_ok = True)
    # Set the value nnUNet_results enviroment variable
    os.environ.setdefault('nnUNet_results', join(mainResults_folder,taskName))

    grouped_rows=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
    # with mp.Pool(processes = 1) as pool:
        @curry  
        def pmap(fun,iterable):
            return pool.map(fun,iterable)

        grouped_rows= toolz.pipe(sourceFrame.iterrows()
                                ,filter(lambda row: row[1]['series_desc'] in modalities_of_intrest)
                                ,groupByMaster
                                ,pmap(partial(iterGroupModalities,modalities_of_intrest=modalities_of_intrest,label_cols=label_cols))
                                ,filter(lambda group: ' ' not in group[1].keys() )
                                ,list
                                ,pmap(partial(add_files,main_modality=main_modality,modalities_of_intrest=modalities_of_intrest,reg_prop=reg_prop,
                                              elacticPath=elacticPath,transformix_path=transformix_path,labelsTrFolder=labelsTrFolder,imagesTrFolder=imagesTrFolder
                                               ,process_labels=process_labels ))
                                ,list
                                )

    data = { 
    "channel_names": channel_names, 
    "labels": label_names, 
    "numTraining": len(grouped_rows), 
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"  # optional! If not provided nnU-Net will automatically determine the ReaderWriter

    }
    # .dumps() as a string
    json_string = json.dumps(data)
    with open(json_path, 'w') as outfile:
        outfile.write(json_string)

    cmd_terminal=f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    p = Popen(cmd_terminal, shell=True)
    p.wait()


    return grouped_rows




# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']
label_cols=lesion_cols
channel_names={  
    "0": "T2", 
    "1": "ADC",
    "2": "HBV",
    "3": "prostate"
    }
label_names= {  # THIS IS DIFFERENT NOW!
    "background": 0,
    "lesion": 1,
    }

grouped_rows= main_prepare_nnunet('279',modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim)
    

group = grouped_rows[5]


print(group)

# nnUNetv2_plan_and_preprocess -d 279 --verify_dataset_integrity
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 279 3d_fullres 0

