"""
segmenation where target is the sum of t2w,adc,hbv labels and we add as input additionally whole prostate segmentations

"""

import SimpleITK as sitk
# import mdai
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import multiprocessing as mp
import functools
from functools import partial
# import mdai
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

# import elastixRegister as elastixRegister
# from elastixRegister import *
from datetime import date
from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import os
from subprocess import Popen
import subprocess

import importlib.util
import importlib
import sys

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res
    
prepareNNunet=loadLib("prepareNNunet","/workspaces/konwersjaJsonData/nnunet/prepareNNunet.py")
from prepareNNunet import *
from scipy import ndimage
import seaborn as sns
import einops
import matplotlib.pyplot as plt
import lapgm

lapgm.use_gpu(True)


debias_obj = lapgm.LapGM(downscale_factor=1)

resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'



# taken from https://github.com/lucianoAvinas/lapgm/blob/main/examples/image_correction.ipynb
# -- Specify hyperparameters --
# tau is an inverse regularization strength on the estimated bias gradient
# n_classes specifies the number of tissue classes visible (including ambient space)
# log_initialize initializes clusters with log transform. Increases sensitivity to finding clusters
#     of similar intensity. For less dispersed bias fields it may be better to set to true.
debias_obj.set_hyperparameters(tau=5e-5, n_classes=6, log_initialize=False) #krowa
# debias_obj.set_hyperparameters(tau=float(os.getenv('tau')), n_classes=int(os.getenv('n_classes')), log_initialize=int(os.getenv('log_initialize'))==1) #krowa



sourceFrame = pd.read_csv(resCSVDir) 
cols=sourceFrame.columns

# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']
prostate_col= 'pg_noSeg' # name of the column with segmentaton of whole prostate gland
new_col_name= 'inferred_pg'
new_col_parts_name='pg_parts_inferred'

non_mri_inputs=[new_col_name,"pz_noSeg",'tz_noSeg']

channel_names={  
    "0": "adc",
    "1": "hbv",
    "2": "t2w",
    "3": "pz_noSeg",
    "4":'tz_noSeg',
    "5":new_col_name
    
    }
# label_names= {  
#     "background": 0,
#     "all": [1,2],
#     }
label_names= {  
    "background": 0,
    "bigger": 1,
    "center": 2
    # "all": [1,2]
    }

def my_crop(image,min_z,min_y,min_x,max_z,max_x,max_y):
    size=[int(max_y-min_y)-1,int(max_x-min_x)-1,int(max_z-min_z)-1]
    beg=[int(min_y),int(min_x),int(min_z)]
    # size=[int(max_x-min_x)-1,int(max_y-min_y)-1,int(max_z-min_z)-1]
    # beg=[int(min_x), int(min_y),int(min_z)]
    extract = sitk.ExtractImageFilter()
    extract.SetSize(size)
    extract.SetIndex(beg)
    extracted_image = extract.Execute(image)
    return extracted_image

def my_concat(grouped):
    # grouped= list(map(lambda tupl: tupl[1],grouped))
    res=np.stack(grouped).astype(int)
    res=np.sum(res,axis=0)
    return res

def get_modalities_to_norm(norm_str, t2w_image,adc_image,hbv_image):
    if(norm_str=="t2w_adc_hbv"):
        return [t2w_image, adc_image, hbv_image]
    if(norm_str=="t2w_adc"):
        return [t2w_image, adc_image]
    if(norm_str=="t2w_hbv"):
        return [t2w_image,  hbv_image]
    if(norm_str=="t2w"):
        return [t2w_image]

def get_im_from_array(arr,channel,orig_im):
    arr=arr[channel,:,:,:]
    image = sitk.GetImageFromArray(arr)  
    image.SetSpacing(orig_im.GetSpacing())
    image.SetOrigin(orig_im.GetOrigin())
    image.SetDirection(orig_im.GetDirection())    
    return image


def return_corrected(norm_str,arrrr,t2w_image,adc_image,hbv_image):
    if(norm_str=="t2w_adc_hbv"):
        return [get_im_from_array(arrrr,0,t2w_image), get_im_from_array(arrrr,1,adc_image), get_im_from_array(arrrr,2,hbv_image)]
    if(norm_str=="t2w_adc"):
        return [get_im_from_array(arrrr,0,t2w_image), get_im_from_array(arrrr,1,adc_image),hbv_image]
    if(norm_str=="t2w_hbv"):
        return [get_im_from_array(arrrr,0,t2w_image), adc_image, get_im_from_array(arrrr,1,hbv_image)]
    if(norm_str=="t2w"):
        return [get_im_from_array(arrrr,0,t2w_image),adc_image, hbv_image]



### bias field correction and normalization
#on the basis of https://github.com/lucianoAvinas/lapgm/blob/main/examples/image_correction.ipynb
def bias_field_and_normalize(t2w_image,adc_image,hbv_image):
    # Approximate location of farthest peak for true data.
    # In practice this can be set to any fixed value of choice.
    TRGT = 0.6
    #first bias field correction
    modalities_to_normalize=  [t2w_image,adc_image,hbv_image]
    modalities_to_normalize = list(map(sitk.GetArrayFromImage ,modalities_to_normalize))
    arrrr = lapgm.to_sequence_array(modalities_to_normalize)
    # Run debias procedure and take parameter output
    params = debias_obj.estimate_parameters(arrrr, print_tols=True)
    arrrr= lapgm.debias(arrrr, params)
    # print(f"ppppp params {params.shape}")
    modalities_to_normalize=  get_modalities_to_norm(os.getenv('to_include_normalize'), arrrr[0,:,:,:],arrrr[1,:,:,:],arrrr[2,:,:,:]) 
    arrrr = lapgm.to_sequence_array(modalities_to_normalize)
    arrrr = lapgm.normalize(brainweb_deb_ex0, params_ex0, target_intensity=TRGT)
    return return_corrected(norm_str,arrrr,t2w_image,adc_image,hbv_image)

def reg_a_to_b_by_metadata_single_d(fixed_image_path,moving_image_path,interpolator):
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0)
    return resampled

def reg_a_to_b_by_metadata_single_c(fixed_image_path,moving_image_path,interpolator):
    # print(f"fixed_image_path {fixed_image_path} moving_image_path {moving_image_path}")
    # moving_image_path=moving_image_path[0]
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)

    # fixed_image=sitk.Cast(fixed_image, sitk.sitkUInt8)
    # moving_image=sitk.Cast(moving_image, sitk.sitkInt)
    
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0)
    return sitk.GetArrayFromImage(resampled)

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

def add_files_custom(group,main_modality,modalities_of_intrest,non_mri_inputs,labelsTrFolder,imagesTrFolder,out_folder):
    # print(f"ggggg {group[1]}")
    if('inferred_pg' not in group[1]):
        print(f"nnnnno inferred_pg")
        return ' '
    if("pz_noSeg" not in group[1]):
        print(f"nnnnno pz_noSeg")
        return ' '    
    if("tz_noSeg" not in group[1]):
        print(f"nnnnno tz_noSeg")
        return ' '    

        
    
    modalit_path_add= list(map( lambda el:(group[1][el]) ,non_mri_inputs))
    # print(f"ggg group[1] {group[1]}")
    # print(f"modalit_path_add[0][1] {modalit_path_add[0][1]}")
    if(modalit_path_add[0][1][0]==' '):
        print(f"no prostate! {group[0]}")
        return ' '


    # modalities_of_intrest_without_main=non_mri_inputs+modalities_of_intrest_without_main

    sources_dict=group[1]
    sources_dict[non_mri_inputs[0]]=(modalit_path_add[0][1][0],)
    sources_dict[non_mri_inputs[1]]=(modalit_path_add[1][1][0],)
    sources_dict[non_mri_inputs[2]]=(modalit_path_add[2][1][0],)


    #http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/20_Expand_With_Interpolators.html
    # print(f"aaaaaaa d  main {sources_dict[main_modality][0]} adc {sources_dict['adc'][0]}")

    if(sources_dict['pz_noSeg'][0]==" "):
        return " "
    if(sources_dict['tz_noSeg'][0]==" "):
        return " "    
    # print(f"ssss sources_dict {sources_dict.keys()}")
    adc_image =reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict['adc'][0], sitk.sitkBSpline)                                 
    hbv_image =reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict['hbv'][0], sitk.sitkBSpline) 

    pz_image =reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict['pz_noSeg'][0], sitk.sitkNearestNeighbor)                                 
    tz_image =reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict['tz_noSeg'][0], sitk.sitkNearestNeighbor)                                 
    try:                                              
        registered_prostate= reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict[new_col_name][0], sitk.sitkNearestNeighbor)
    except:
        print(f"nooo registered prostate !!")
        return " "
       
    prostate_arr= reg_a_to_b_by_metadata_single_c(sources_dict[main_modality][0],sources_dict[new_col_name][0], sitk.sitkNearestNeighbor)


    ########### manage labels

    labels_hbv=group[1]['hbv'][1]
    labels_adc=group[1]['adc'][1]


    labels_hbv=list(map(lambda pathh_moving: reg_a_to_b_by_metadata_single_b(sources_dict[main_modality][0],pathh_moving,out_folder,sitk.sitkNearestNeighbor)                                    
                                                      ,labels_hbv ))
    labels_adc=list(map(lambda pathh_moving: reg_a_to_b_by_metadata_single_b(sources_dict[main_modality][0],pathh_moving,out_folder,sitk.sitkNearestNeighbor)                                    
                                                      ,labels_adc ))
    labels= labels_adc+labels_hbv
    labRes= np.zeros_like(prostate_arr)
    if(len(labels)>0):

        label_names=list(map(lambda pathh:  
                            list(filter( lambda el_out:'/' not in el_out ,
                                list(filter(lambda el: 'lesion' in el  ,pathh.split('_')))))[0]
                            
                            ,labels ))

        label_names= list(filter( lambda el: '/' not in el ,label_names))
        # we want to get sum of all labels of the same name - so lesion 1 sum lesion 2 sum ...
        # then we encode it in separate file 
        zipped_names= list(zip(label_names,labels))
        grouped_by_lesion_num = dict(groupby(lambda tupl :tupl[0],zipped_names)).items()
        grouped_by_lesion_num= list(map(lambda grouped :  (grouped[0],list(map(lambda el: el[1],grouped[1])) ) , grouped_by_lesion_num  ))
        # print(f"\n grouped_by_lesion_num {grouped_by_lesion_num} \n")

        # we need to get the paths to bool arrs and sum it all
        # next we keep all >1
        # print(f"grouped_by_lesion_num {grouped_by_lesion_num}")
        grouped_by_lesion_num_arrs= list(map(lambda grouped : 
                                              list(map(get_bool_arr_from_path, grouped[1]))
                                                ,grouped_by_lesion_num  ))

        grouped_by_lesion_num_arrs= list(map(my_concat,grouped_by_lesion_num_arrs  ))

        grouped_by_lesion_num_arrs= np.stack(grouped_by_lesion_num_arrs,axis=0).astype(int)

        reduced_common= (np.sum(grouped_by_lesion_num_arrs,axis=0)>1).astype(int)

        reduced_sum = np.array(functools.reduce(get_bool_or, labels)).astype(bool)
        # reduced_sum= ndimage.binary_dilation(reduced_sum,iterations=2)

        reduced_common_eroded= ndimage.binary_erosion(reduced_common,iterations=1)
        summ=np.sum(reduced_common.flatten())
        if(summ>0):
            ratio=(np.sum(reduced_common_eroded.flatten()) /np.sum(reduced_common.flatten()))
            print(f"ratio {ratio}")
            if(ratio>0.15 ):
                reduced_common=reduced_common_eroded

        # reduced_common_eroded= ndimage.binary_erosion(reduced_common,iterations=1)
        # if(summ>0):
        #     ratio=(np.sum(reduced_common_eroded.flatten()) /np.sum(reduced_common.flatten()))
        #     # print(f"ratio {ratio}")
        #     if(ratio>0.15 ):
        #         reduced_common=reduced_common_eroded


        labRes=reduced_sum#(reduced_sum>0).astype(int)
        labRes=(labRes+(reduced_common.astype(np.uint8))).astype(np.uint8)

        







    prostate_arr_indicies= np.argwhere(prostate_arr)



    add_padd_z=50
    add_padd_x=14
    add_padd_y=14

    max_z= np.max(prostate_arr_indicies[:,0])+add_padd_z
    min_z= np.min(prostate_arr_indicies[:,0])-add_padd_z
    # min_z=0
    # max_z=adc_arr.shape[0]
    max_x= np.max(prostate_arr_indicies[:,1])+add_padd_x
    min_x= np.min(prostate_arr_indicies[:,1])-add_padd_x

    max_y= np.max(prostate_arr_indicies[:,2])+add_padd_y
    min_y= np.min(prostate_arr_indicies[:,2])-add_padd_y
    
    min_z=max(min_z,0)
    min_y=max(min_y,0)
    min_x=max(min_x,0)

    max_z=min(max_z,prostate_arr.shape[0]-1)
    max_x=min(max_x,prostate_arr.shape[1]-1)
    max_y=min(max_y,prostate_arr.shape[2]-1)


    # min_x=0
    # max_x=adc_arr.shape[1]

    # min_y=0
    # max_y=adc_arr.shape[2]
    t2w_image = sitk.ReadImage(group[1][main_modality][0])
    
    # inferred_parts_image= sitk.ReadImage(sources_dict[new_col_parts_name][0])

    # inferred_parts_image_arr=sitk.GetArrayFromImage(inferred_parts_image)

    print(f"labRes 1 {np.sum((labRes==1).flatten())}  labRes 2 {np.sum((labRes==2).flatten())}")
    label_image = get_from_arr(labRes,t2w_image)

    # pz_arrr=(inferred_parts_image_arr==1)
    # tz_arrr=(inferred_parts_image_arr==2)
    

    # pz_image = get_from_arr(pz_arrr.astype(int),t2w_image)
    # tz_image = get_from_arr((inferred_parts_image_arr==2).astype(int),t2w_image)



    adc_image=my_crop(adc_image,min_z,min_y,min_x,max_z,max_x,max_y)
    hbv_image=my_crop(hbv_image,min_z,min_y,min_x,max_z,max_x,max_y)
    t2w_image=my_crop(t2w_image,min_z,min_y,min_x,max_z,max_x,max_y)    
    pz_image=my_crop(pz_image,min_z,min_y,min_x,max_z,max_x,max_y)
    tz_image=my_crop(tz_image,min_z,min_y,min_x,max_z,max_x,max_y)
    




    
    
    registered_prostate=my_crop(registered_prostate,min_z,min_y,min_x,max_z,max_x,max_y)
    
    
    label_image=my_crop(label_image,min_z,min_y,min_x,max_z,max_x,max_y)

    
    label_new_path,out_pathsDict=prepare_out_paths(group,modalities_of_intrest,labelsTrFolder,imagesTrFolder,non_mri_inputs,channel_names )

    # print(f"ooo out_pathsDict {out_pathsDict}")

    writer = sitk.ImageFileWriter()
    writer.SetFileName(label_new_path)
    writer.Execute(label_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_pathsDict['adc'])
    writer.Execute(adc_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_pathsDict['hbv'])
    writer.Execute(hbv_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_pathsDict['t2w'])
    writer.Execute(t2w_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_pathsDict["pz_noSeg"])
    writer.Execute(pz_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_pathsDict['tz_noSeg'])
    writer.Execute(tz_image)

    return group[0],[out_pathsDict['t2w'], out_pathsDict['adc'], out_pathsDict['hbv']]


def to_map_bias_corr_and_norm(tupl):
    print(f"ttttt {tupl}")
    id,pathss=tupl
    t2w_image,adc_image,hbv_image=bias_field_and_normalize(sitk.ReadImage(pathss[0]),sitk.ReadImage(pathss[1]),hbv_sitk.ReadImage(pathss[2]))
    writer = sitk.ImageFileWriter()
    writer.SetFileName(pathss[0])
    writer.Execute(t2w_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(pathss[1])
    writer.Execute(adc_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(pathss[2])
    writer.Execute(hbv_image)


def main_func():
    #metadata directory
    resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
    #directory with inferred prostates
    dir_inferred_prost='/home/sliceruser/workspaces/konwersjaJsonData/my_prost_infered'
    # dir_inferred_prost='/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_infered'
    # dir_inferred_prost='/home/sliceruser/workspaces/konwersjaJsonData/my_prost_infered'
    # dir_inferred_prost_parts='/home/sliceruser/workspaces/konwersjaJsonData/my_prost_parts_infered'



    sourceFrame = pd.read_csv(resCSVDir)


    out_folder='/home/sliceruser/explore/temp'
    os.makedirs(out_folder ,exist_ok = True)
    shutil.rmtree(out_folder)
    os.makedirs(out_folder ,exist_ok = True)


    cols=sourceFrame.columns
    noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
    lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))

    sourceFrame=add_inferred_full_prost_to_dataframe(dir_inferred_prost, sourceFrame,new_col_name,out_folder,"/workspaces/konwersjaJsonData/explore/prost_full.csv")
    sourceFrame=add_inferred_full_prost_to_dataframe(dir_inferred_prost, sourceFrame,new_col_parts_name,out_folder,"/workspaces/konwersjaJsonData/explore/prost_parts.csv")
    sourceFrame['tz_inferred']=' '
    test_ids = pd.read_csv('/workspaces/konwersjaJsonData/test_ids.csv' ).to_numpy().flatten()

    test_ids= list(map(lambda el: str(el).strip(),test_ids ))
    filter_ids=lambda row: str(row[1]['masterolds']).strip() not in test_ids
    cols=sourceFrame.columns
    noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']
    lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))
    main_modality = 't2w'
    dataset_id=101



    # with mp.Pool(processes = mp.cpu_count()) as pool:
    # with mp.Pool(processes = 1) as pool:
    #     @curry  
    #     def pmap(fun,iterable):
    #         return pool.map(fun,iterable)

    nNunetBaseFolder='/home/sliceruser/nnunetMainFolder'

    os.makedirs(f"{nNunetBaseFolder}/nnUNet_preprocessed" ,exist_ok = True)
    os.makedirs(f"{nNunetBaseFolder}/nnUNet_raw" ,exist_ok = True)




    shutil.rmtree(f"{nNunetBaseFolder}/nnUNet_preprocessed")
    shutil.rmtree(f"{nNunetBaseFolder}/nnUNet_raw")

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


    ids=[]
    # with mp.Pool(processes = 1) as pool:
    with mp.Pool(processes = mp.cpu_count()) as pool:
        @curry  
        def pmap(fun,iterable):
            return pool.map(fun,iterable)

        ids=toolz.pipe(sourceFrame.iterrows()
                                        ,filter(lambda row: row[1]['series_desc'] in modalities_of_intrest)
                                        ,filter(filter_ids) # filter out all of the test cases
                                        ,groupByMaster
                                        ,pmap(partial(iterGroupModalities,modalities_of_intrest=modalities_of_intrest,label_cols=lesion_cols,non_mri_inputs=non_mri_inputs))
                                        ,filter(lambda group: ' ' not in group[1].keys() )
                                        ,list
                                        ,pmap(partial(add_files_custom,main_modality=main_modality,modalities_of_intrest=modalities_of_intrest,non_mri_inputs=non_mri_inputs,labelsTrFolder=labelsTrFolder,imagesTrFolder=imagesTrFolder,out_folder=out_folder))                            
                                        ,list
                                        ,filter(lambda el: el!=' ')
                                        ,list
                                        ,map(to_map_bias_corr_and_norm)
                                        ,list)

    channel_names={  
        "1": "noNorm",
        "2": "noNorm",
        "3": "zscore",
        "4": "noNorm",
        "5": "noNorm"

        }
    #https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_plans_files.md
    data = { 
        "channel_names": channel_names, 
        "labels": label_names,  
        # "regions_class_order": [2,1,0],  
        "file_ending": ".nii.gz",
        "numTraining" : len(ids),
        
        "nnUNetPlans" : ['2d','3d_fullres'],
        
        # 'conv_kernel_sizes': [[1, 3, 3], [1, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
    }

    print(f"lllllllllllllllllllllllllll {len(ids)}")

    json_string = json.dumps(data)
    with open(json_path, 'w') as outfile:
        outfile.write(json_string)



    cmd_terminal=f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    p = Popen(cmd_terminal, shell=True)
    p.wait()



    # plans_path= join(preprocesss_folder,taskName,'nnUNetPlans.json')
    plans_path= f"/home/sliceruser/nnunetMainFolder/nnUNet_preprocessed/Dataset{dataset_id}_Prostate/nnUNetPlans.json"
    f = open(plans_path)
    plans = json.load(f)
    plans['configurations']['3d_lowres'] = {
        "data_identifier": "nnUNetPlans_3d_lowres",  # do not be a dumbo and forget this. I was a dumbo. And I paid dearly with ~10 min debugging time
        'inherits_from': '3d_fullres',
    'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2
    #    , 'patch_size': [32, 96, 96]
    #     #   'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2, 'patch_size': [96, 96, 96] # for swin
    #                                                 , 'median_image_size_in_voxels': [32., 84., 95.]
    # , 'spacing': [0.78125, 0.78125   , 0.78125   ] #for swin
    # # , 'spacing': [3.30000019, 0.78125   , 0.78125   ]

    # , 'normalization_schemes': ['NoNormalization', 'NoNormalization', 'ZScoreNormalization', 'NoNormalization', 'NoNormalization']
    # , 'use_mask_for_norm': [False, False, False, False, False]
    # , 'UNet_class_name': 'PlainConvUNet'
    , 'UNet_base_num_features': 180
    }

    # 3D fullres U-Net configuration:
    # {'data_identifier': 'nnUNetPlans_3d_fullres', 'preprocessor_name': 'DefaultPreprocessor'
    # , 'batch_size': 10, 'patch_size': array([40, 96, 96]), 'median_image_size_in_voxels': array([40., 84., 95.])
    # , 'spacing': array([3.30000019, 0.78125   , 0.78125   ])
    # , 'normalization_schemes': ['NoNormalization', 'NoNormalization', 'ZScoreNormalization', 'NoNormalization', 'NoNormalization']
    # , 'use_mask_for_norm': [False, False, False, False, False], 'UNet_class_name': 'PlainConvUNet'
    # , 'UNet_base_num_features': 32, 'n_conv_per_stage_encoder': (2, 2, 2, 2, 2), 'n_conv_per_stage_decoder': (2, 2, 2, 2)
    # , 'num_pool_per_axis': [2, 4, 4], 'pool_op_kernel_sizes': [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
    # , 'conv_kernel_sizes': [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'unet_max_num_features': 320
    # , 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape'
    # , 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}
    # , 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}
    # , 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape'
    # , 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0
    # , 'force_separate_z': None}, 'batch_dice': False}


    json_string = json.dumps(plans)     
    print(f"aaaaaaaaaaaaa {json_string} ")
        
    with open(plans_path, 'w') as outfile:
        outfile.write(json_string)

    data = { 
        "channel_names": channel_names, 
        "labels": label_names,  
        # "regions_class_order": [2,1,0],  
        "file_ending": ".nii.gz",
        "numTraining" : len(ids),
        
        # "nnUNetPlans" : ['2d','3d_fullres','3d_fullres_custom','3d_lowres','3d_cascade_fullres'],
        "nnUNetPlans" : ['2d','3d_fullres','3d_fullres_custom'],
        
        # 'conv_kernel_sizes': [[1, 3, 3], [1, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
    }


    json_string = json.dumps(data)
    with open(json_path, 'w') as outfile:
        outfile.write(json_string)
        


    cmd_terminal=f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    p = Popen(cmd_terminal, shell=True)
    p.wait()

# main_func()

# cp -a /home/sliceruser/nnunetMainFolder/nnUNet_raw/Dataset101_Prostate/imagesTr /workspaces/konwersjaJsonData/explore/preprocessed
# cp -a /home/sliceruser/nnunetMainFolder/nnUNet_raw/Dataset101_Prostate/labelsTr /workspaces/konwersjaJsonData/explore/preprocessed_labels


#### image_processing_oneformer is modified


# my_proj_name="seg lesions 4" tag="l4d  classic nnunet" my_proj_desc="" nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl
# my_proj_name="seg lesions debug" tag="l4a test" my_proj_desc="l4a test" nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl


#with masked binary_cross_entropy_with_logits
# 2023-08-05 17:35:45.361709: train_loss 0.0002
# 2023-08-05 17:35:45.362278: val_loss 0.0003
# 2023-08-05 17:35:45.362691: Pseudo dice [0.0144, 0.2775]
# 2023-08-05 17:35:45.363019: Percent in [0.5436]
# 2023-08-05 17:35:45.363312: Percent out [0.101]
# 2023-08-05 17:35:45.363678: Epoch time: 202.81 s
# 2023-08-05 17:36:00.034846: 
# 2023-08-05 17:36:00.035831: Epoch 88

#focal +cross entropy loss
# 2023-08-05 23:57:08.044413: Current learning rate: 0.00921
# 2023-08-06 00:00:23.126401: train_loss 0.0018
# 2023-08-06 00:00:23.127024: val_loss 0.0028
# 2023-08-06 00:00:23.127418: Pseudo dice [0.0039, 0.322]
# 2023-08-06 00:00:23.127787: Percent in [0.6303]
# 2023-08-06 00:00:23.128125: Percent out [0.3697]
# 2023-08-06 00:00:23.128488: Epoch time: 195.11 s
# 2023-08-06 00:00:23.129513: Yayy! New best EMA pseudo Dice: [0.629]
# 2023-08-06 00:00:36.670725: 
# 2023-08-06 00:00:36.671606: Epoch 88


# cd /home/sliceruser/nnunetMainFolder/nnUNet_raw/Dataset101_Prostate/imagesTr
# cp 9334200_0000.nii.gz /workspaces/konwersjaJsonData/explore/temp/9334200_0000.nii.gz
# cp 9334200_0001.nii.gz /workspaces/konwersjaJsonData/explore/temp/9334200_0001.nii.gz
# cp 9334200_0002.nii.gz /workspaces/konwersjaJsonData/explore/temp/9334200_0002.nii.gz


# /home/sliceruser/nnunetMainFolder/nnUNet_preprocessed/Dataset101_Prostate/3d_fullres_custom