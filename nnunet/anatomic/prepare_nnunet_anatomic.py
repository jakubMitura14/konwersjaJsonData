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

# from elastixRegister import reg_a_to_b,reg_a_to_b_be_meta_data,reg_a_to_b_by_metadata_single_b
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
import os
import subprocess
from pathlib import Path

import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)


elacticPath='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/elastix'
transformix_path='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/transformix'
reg_prop='/workspaces/konwersjaJsonData/nnunet/registration/parameters.txt'  
# dataframe with master ids that we should not include in training
test_ids_CSVDir='/workspaces/konwersjaJsonData/test_ids.csv'
test_ids=pd.read_csv(test_ids_CSVDir)['ids'].to_numpy().flatten()



def reg_a_to_b_by_metadata_single_b(fixed_image_path,moving_image_path,out_folder, interpolator=sitk.sitkNearestNeighbor):
    if(len(moving_image_path)<4):
        moving_image_path=moving_image_path[0]
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)

    # fixed_image=sitk.Cast(fixed_image, sitk.sitkUInt8)
    # moving_image=sitk.Cast(moving_image, sitk.sitkInt)
    
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0)
    
    # print(f" prim sum {np.sum(sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path)).flatten())} \n suuum {np.sum(sitk.GetArrayFromImage(resampled).flatten())} ")
  
    writer = sitk.ImageFileWriter()
    new_path= join(out_folder,moving_image_path.split('/')[-1])
    writer.SetFileName(new_path)
    writer.Execute(resampled)

    return new_path

def reg_a_to_b_by_metadata_single_f(fixed_image_path,moving_image_path,interpolator):

    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0)
    return resampled



def reg_a_to_b_by_metadata_single_g(fixed_image_path,moving_image_path,out_folder, interpolator=sitk.sitkNearestNeighbor):
    if(len(moving_image_path)<4):
        moving_image_path=moving_image_path[0]
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)

    # fixed_image=sitk.Cast(fixed_image, sitk.sitkUInt8)
    # moving_image=sitk.Cast(moving_image, sitk.sitkInt)
    
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0)
    
    # print(f" prim sum {np.sum(sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path)).flatten())} \n suuum {np.sum(sitk.GetArrayFromImage(resampled).flatten())} ")
  
    writer = sitk.ImageFileWriter()
    new_path= moving_image_path
    writer.SetFileName(new_path)
    writer.Execute(resampled)

    return new_path


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


def get_bool_and(pathA,pathB):
    if(isinstance(pathA, str) and isinstance(pathB, str)):
        return np.logical_and(get_bool_arr_from_path(pathA),get_bool_arr_from_path(pathB))
    elif(isinstance(pathA, str) and not isinstance(pathB, str)):
        return np.logical_and(get_bool_arr_from_path(pathA),pathB)
    elif(not isinstance(pathA, str) and isinstance(pathB, str)):
        return np.logical_and(get_bool_arr_from_path(pathB),pathA)
    else:
        return np.logical_and(pathB,pathA)

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


def get_from_arr(zeroArray,image3D):
    """
    given array saves it to file into defined path using simpleitk
    """
    image = sitk.GetImageFromArray(zeroArray.astype(float).astype(np.uint8))  
    # nan_count=np.sum(np.isnan(np.array(sitk.GetArrayFromImage(image)).flatten()))
    # if(nan_count>0):
    #     raise ValueError(f"!!! nan in image would be saved as {newPathLab}")

    image.SetSpacing(image3D.GetSpacing())
    image.SetOrigin(image3D.GetOrigin())
    image.SetDirection(image3D.GetDirection())   
    image = sitk.DICOMOrient(image, 'LPS')
    return image


def save_from_arr(zeroArray,image3D,newPathLab):
    """
    given array saves it to file into defined path using simpleitk
    """
    writer = sitk.ImageFileWriter()
    image = get_from_arr(zeroArray,image3D)
    # image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    writer.SetFileName(newPathLab)
    writer.Execute(image)
    return newPathLab

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






# def set_common_prhysical_data():

#     image_out.SetOrigin(img.GetOrigin())
#     image_out.SetSpacing(img.GetSpacing())
#     #set to RAI
#     image_out.SetDirection(tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

def write_file(image,outputImageFileName):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)

def add_files(group,main_modality,modalities_of_intrest,reg_prop,elacticPath,transformix_path,labelsTrFolder
              ,imagesTrFolder,process_labels,non_mri_inputs,channel_names,is_to_preprocess ):
    """
    first register images and their respective labels to t2w
    then reduces all labels into their sum
    then saves mri and reduced labels into nnunet workdir to get structure the same as in baseline picai nnunet algorithm
    """
    modalit_path_add= list(map( lambda el:(group[1][el]) ,non_mri_inputs))
    filtered= list(filter(lambda el: el==' ',modalit_path_add))

    if len(filtered)!=0:
        return (' ',{})
    
    
    label_new_path,out_pathsDict=prepare_out_paths(group,modalities_of_intrest,labelsTrFolder,imagesTrFolder,non_mri_inputs,channel_names )
    label_new_path_prim=label_new_path
    temp_dir = tempfile.mkdtemp() # temporary directory
    modalities_of_intrest_without_main= list(filter( lambda el: el!=main_modality , modalities_of_intrest))
    modalities=[]
    labels=[]
    mris=[]
    newPaths=[]



    modalities.append(main_modality)
    # zipped_modalit_path = list(zip(modalities,mris))
    

    mris=list(map(lambda mod:   reg_a_to_b_by_metadata_single_b(group[1][main_modality][0],group[1][mod][0],temp_dir, sitk.sitkBSpline)
                 ,modalities_of_intrest_without_main))


    mris=list(mris)
    # mris.append(group[1][main_modality][0])    
    mris.append(group[1][main_modality][0])    
    modalities_of_intrest_without_main.append(main_modality)
    modalities=modalities_of_intrest_without_main

    #adding to the list the labels from main modality thay did not needed to be registered
    labels=np.array(labels+group[1][main_modality][1]).flatten()
    # print(f"labels {labels}  group[1][main_modality] {group[1][main_modality]} ")
    example_scan=[]
    if(len(labels)>0):
        zipped_modalit_path = list(zip(modalities,mris))
        # print(f"zipped_modalit_path {zipped_modalit_path}")
        #zipping for starmap use        
        zipped_modalit_path= list(map( lambda tupl:(tupl[1], out_pathsDict[tupl[0]]) ,zipped_modalit_path))
        # print(f"mmmm mris {mris} modalities_of_intrest {modalities_of_intrest}")
        if(is_to_preprocess):
            mris_for_sample=[mris[2],mris[0],mris[1]]
            sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in mris_for_sample
            ],
            settings=PreprocessingSettings(
                physical_size=[81.0, 192.0, 192.0],
                crop_only=True
            )
            )

            # perform preprocessing
            sample.preprocess()
            # print(f"ssss sample.scans {sample.scans}")
            out=[zipped_modalit_path[2][1],zipped_modalit_path[0][1],zipped_modalit_path[1][1]]
            list(itertools.starmap(write_file,list(zip(sample.scans, out ))))
            example_scan=zipped_modalit_path[2][1]

        if(len(non_mri_inputs)>0):
            prost_im=reg_a_to_b_by_metadata_single_f(example_scan,group[1][non_mri_inputs[0]][1][0],sitk.sitkNearestNeighbor)
            sitk.WriteImage(prost_im, out_pathsDict[non_mri_inputs[0]])

        # # write preprocessed scans to nnUNet input directory
        # for i, scan in enumerate(sample.scans):
        #     path = self.nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
        #     atomic_image_write(scan, path)
        

        
        
        
        # zipped_modalit_path_add= list(map( lambda el:(group[1][el][1][0], out_pathsDict[el]) ,non_mri_inputs))
        
        
        # # zipped_modalit_path=zipped_modalit_path+zipped_modalit_path_add
        # zipped_modalit_path=zipped_modalit_path_add
        # # print(f"zipped_modalit_path_add {zipped_modalit_path_add} \n  ")
        # # print(f"zipped_modalit_path_add {zipped_modalit_path_add} \n zipped_modalit_path {zipped_modalit_path} \n out_pathsDict {out_pathsDict}  ")
        
        # zipped_modalit_path= list(filter(  lambda tupl: tupl[0]!=" " and tupl[1]!=" ",zipped_modalit_path))
        
        
        #as we already have prepared the destination paths and sources for images we need now to copy files
        # we need to remember that we are  getting from mha to nii gz


        # list(itertools.starmap(copy_changing_type ,zipped_modalit_path ))
        # list(itertools.starmap(copy_changing_type ,zipped_modalit_path_add ))
        _,new_mri_paths= list(toolz.sandbox.core.unzip(zipped_modalit_path))
        new_mri_paths=np.unique(list(new_mri_paths)+non_mri_inputs)
        
        newPaths= list(zip(modalities,new_mri_paths))
        non_mri_inputs_new_paths= list(map( lambda el:(el, out_pathsDict[el]) ,non_mri_inputs))

        # print(f"aaaaa non_mri_inputs {non_mri_inputs} \n non_mri_inputs_new_paths {non_mri_inputs_new_paths} \n")

        newPaths=newPaths+non_mri_inputs_new_paths

        
 
        label_new_paths,newPaths= process_labels(labels,group,main_modality,label_new_path,newPaths,out_pathsDict)
        if(is_to_preprocess):
            reg_a_to_b_by_metadata_single_g(example_scan,label_new_path_prim,sitk.sitkNearestNeighbor)
        
        for label_new_path in label_new_paths:
            newPaths.append(('label',label_new_path ))
        #copying label holding segmentation of full prostate gland
        # currProstPath= group[1]['prostate']
        # shutil.copyfile(currProstPath,out_prostate_path )
        # newPaths.append(('prostate',out_prostate_path))

        #clearing temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        newPaths_paths= list(map(lambda tupl: tupl[1], newPaths))

        return label_new_paths #(group[0],dict(newPaths))
    return " "

def main_prepare_nnunet(dataset_id, modalities_of_intrest,channel_names,label_names,label_cols,process_labels,non_mri_inputs
                        ,sourceFrame,main_modality,for_filter_unwanted=None,is_test_prep=False,is_to_preprocess=True,generate_plans=True):
    """
    main function for preparing nnunet
    """
    #first removing old data
    nNunetBaseFolder='/home/sliceruser/nnunetMainFolder'

    shutil.rmtree(f"{nNunetBaseFolder}/nnUNet_preprocessed")
    shutil.rmtree(f"{nNunetBaseFolder}/nnUNet_raw")

    print(f"aaaaaaaa {nNunetBaseFolder}/nnUNet_raw")

    taskName= f"Dataset{dataset_id}_Prostate"
    taskFolder = join(nNunetBaseFolder,'nnUNet_raw',taskName)
    preprocesss_folder= join(nNunetBaseFolder,'nnUNet_preprocessed')
    results_folder= join(nNunetBaseFolder,'nnUNet_results')
    mainResults_folder="/home/sliceruser/nnUNet_results"
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
    
    
        #plans well explained in https://github.com/MIC-DKFZ/nnUNet/blob/4612d35b1b80d558f3e1a650decd408f6f70c68a/documentation/explanation_plans_files.md?plain=1#L164

    
    
    
    #      "configurations": {
    #   "3d_lowres": {
    #     "inherits_from": "3d_fullres",
    #     "data_identifier": "3d_lowres",
    #     "spacing": [2.0, 2.0, 2.0], # from [1.0, 1.0, 1.0] in 3d_fullres
    #     "median_image_size_in_voxels": [18, 25, 18], # from [36, 50, 35]
    #     "patch_size": [20, 28, 20], # from [40, 56, 40]
    #     "n_conv_per_stage_encoder": [2, 2, 2], # one less entry than 3d_fullres ([2, 2, 2, 2])
    #     "n_conv_per_stage_decoder": [2, 2], # one less entry than 3d_fullres
    #     "num_pool_per_axis": [2, 2, 2], # one less pooling than 3d_fullres in each dimension (3d_fullres: [3, 3, 3])
    #     "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]], # one less [2, 2, 2]
    #     "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]], # one less [3, 3, 3]
    #     "next_stage": "3d_cascade_fullres" # name of the next stage in the cascade
    #   },
    #   "3d_cascade_fullres": { # does not need a data_identifier because we can use the data of 3d_fullres
    #     "inherits_from": "3d_fullres",
    #     "previous_stage": "3d_lowres" # name of the previous stage
    #   }
    

    
    
    # Set the value nnUNet_results enviroment variable
    # os.environ.setdefault('nnUNet_results', join(mainResults_folder,taskName))
    #defaul is not filter out anything
    if(for_filter_unwanted==None):
        for_filter_unwanted=lambda group: True
    grouped_rows=[]
    
    filter_out_test_ids=lambda row: row[1]['masterolds'] not in test_ids
    filter_in_test_ids=lambda row: row[1]['masterolds'] in test_ids
    filter_ids=filter_out_test_ids
    if(is_test_prep):
        filter_ids=filter_in_test_ids
    
    with mp.Pool(processes = mp.cpu_count()) as pool:
    # with mp.Pool(processes = 1) as pool:
        @curry  
        def pmap(fun,iterable):
            return pool.map(fun,iterable)
        label_paths= toolz.pipe(sourceFrame.iterrows()
                                ,filter(lambda row: row[1]['series_desc'] in modalities_of_intrest)
                                ,filter(filter_ids) # filter out all of the test cases
                                ,groupByMaster
                                ,pmap(partial(iterGroupModalities,modalities_of_intrest=modalities_of_intrest,label_cols=label_cols,non_mri_inputs=non_mri_inputs))
                                ,filter(lambda group: ' ' not in group[1].keys() )
                                ,filter(for_filter_unwanted )
                                ,list
                                ,pmap(partial(add_files,main_modality=main_modality,modalities_of_intrest=modalities_of_intrest,reg_prop=reg_prop,
                                              elacticPath=elacticPath,transformix_path=transformix_path,labelsTrFolder=labelsTrFolder,imagesTrFolder=imagesTrFolder
                                               ,process_labels=process_labels,non_mri_inputs=non_mri_inputs,channel_names=channel_names,is_to_preprocess=is_to_preprocess ))
                                ,filter(lambda el: el!=' ')                                
                                ,list
                                )

    label_paths= list(itertools.chain(*label_paths))
    print(f"llllllllabel_paths {len(label_paths)}")
    # label_paths= list(itertools.chain(*label_paths))

    data = { 
    "channel_names": channel_names, 
    "labels": label_names,  
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
    "regions_class_order": [5,4,3,2,1],  
    "normalization_schemes" : ["zscore","noNorm","noNorm"],
    "numTraining" : len(label_paths),
    "nnUNetPlans" : ['2d','3d_lowres','3d_cascade_fullres', '3d_fullres']
    }

    
    
    # data['configurations']['3d_lowres'] = {
    #         "data_identifier": "nnUNetPlans_3d_lowres",  # do not be a dumbo and forget this. I was a dumbo. And I paid dearly with ~10 min debugging time
    #         'inherits_from': '3d_fullres',
    #         "patch_size": [20, 28, 20],
    #         "median_image_size_in_voxels": [18.0, 25.0, 18.0],
    #         "spacing": [2.0, 2.0, 2.0],
    #         "n_conv_per_stage_encoder": [2, 2, 2],
    #         "n_conv_per_stage_decoder": [2, 2],
    #         "num_pool_per_axis": [2, 2, 2],
    #         "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
    #         "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
    #         "next_stage": "3d_cascade_fullres"
    #     }
    # data['configurations']['3d_cascade_fullres'] = {
    #         'inherits_from': '3d_fullres',
    #         "previous_stage": "3d_lowres"
    #     }
    

    # .dumps() as a string
    json_string = json.dumps(data)
    with open(json_path, 'w') as outfile:
        outfile.write(json_string)
    

    cmd_terminal=f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    print(f"ccccc {cmd_terminal} \n")
    p = Popen(cmd_terminal, shell=True)
    p.wait()



    # plans_path= join(preprocesss_folder,taskName,'nnUNetPlans.json')
    plans_path= f"/home/sliceruser/nnunetMainFolder/nnUNet_preprocessed/Dataset{dataset_id}_Prostate/nnUNetPlans.json"
    f = open(plans_path)
    plans = json.load(f)
    plans['configurations']['3d_lowres'] = {
        "data_identifier": "nnUNetPlans_3d_lowres",  # do not be a dumbo and forget this. I was a dumbo. And I paid dearly with ~10 min debugging time
        'inherits_from': '3d_fullres',
 'preprocessor_name': 'DefaultPreprocessor'
 , 'batch_size': 1, 'patch_size': [ 32, 256, 256], 'median_image_size_in_voxels': [ 32., 246., 246.]
 , 'spacing': [3.29999995, 0.78125   , 0.78125   ]
 , 'normalization_schemes': ['ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization']
 , 'use_mask_for_norm': [False, False, False], 'UNet_class_name': 'PlainConvUNet'
 , 'UNet_base_num_features': 104, 'n_conv_per_stage_encoder': (2, 2, 2, 2, 2, 2, 2)
 , 'n_conv_per_stage_decoder': (2, 2, 2, 2, 2, 2), 'num_pool_per_axis': [2, 6, 6]
 , 'pool_op_kernel_sizes': [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
 , 'conv_kernel_sizes': [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
 , 'unet_max_num_features': 320, 'resampling_fn_data': 'resample_data_or_seg_to_shape'
 , 'resampling_fn_seg': 'resample_data_or_seg_to_shape'
 , 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}
 , 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}
 , 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape'
 , 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'batch_dice': False}


    json_string = json.dumps(plans)     
    print(f"aaaaaaaaaaaaa {json_string} \n \n \n ppppppppppppppppppppppppppppppppppp")
        
    with open(plans_path, 'w') as outfile:
        outfile.write(json_string)

    # data = { 
    #     "channel_names": channel_names, 
    #     "labels": label_names,  
    #     # "regions_class_order": [2,1,0],  
    #     "file_ending": ".nii.gz",
    #     "numTraining" : len(label_paths),
        
    #     # "nnUNetPlans" : ['2d','3d_fullres','3d_fullres_custom','3d_lowres','3d_cascade_fullres'],
    #     "nnUNetPlans" : ['2d','3d_fullres','3d_fullres_custom'],
        
    #     # 'conv_kernel_sizes': [[1, 3, 3], [1, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
    # }


    # json_string = json.dumps(data)
    # with open(json_path, 'w') as outfile:
    #     outfile.write(json_string)
        


    cmd_terminal=f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    p = Popen(cmd_terminal, shell=True)
    p.wait()



    return grouped_rows

#nnUNetv2_train 279 3d_lowres -tr 0




# nnUNetv2_plan_and_preprocess -d 285 --verify_dataset_integrity -c [2d 3d_fullres 3d_lowres]-np 8

# print(np.array(cols))

# print(grouped_rows[0])

# print(len(grouped_rows))

# mapped= list(map(lambda el:len(el[1]),grouped_rows))
# print(f"lensss \n {mapped}")

# nnUNetv2_plan_and_preprocess -d 279 --verify_dataset_integrity
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 279 3d_fullres 0

# 9041700

# ['/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset279_Prostate/imagesTr/9041700_0000.nii.gz', '/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset279_Prostate/imagesTr/9041700_0001.nii.gz', '/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset279_Prostate/imagesTr/9041700_0002.nii.gz', '/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset279_Prostate/imagesTr/9041700_0003.nii.gz', '/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset279_Prostate/labelsTr/9041700.nii.gz'] 