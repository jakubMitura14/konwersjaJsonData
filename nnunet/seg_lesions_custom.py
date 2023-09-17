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

import elastixRegister as elastixRegister
from elastixRegister import *
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
from prepareNNunet import *
from scipy import ndimage
import seaborn as sns
import einops
import matplotlib.pyplot as plt


resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'

sourceFrame = pd.read_csv(resCSVDir) 
cols=sourceFrame.columns

# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']
prostate_col= 'pg_noSeg' # name of the column with segmentaton of whole prostate gland
new_col_name= 'inferred_pg'
new_col_parts_name='pg_parts_inferred'

non_mri_inputs=[new_col_name,new_col_parts_name,'tz_inferred']

channel_names={  
    "0": "adc",
    "1": "hbv",
    "2": "t2w",
    "3": non_mri_inputs[1],
    "4":'tz_inferred',
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

def add_files_custom(group,main_modality,modalities_of_intrest,non_mri_inputs,labelsTrFolder,imagesTrFolder):

    if('inferred_pg' not in group[1]):
        print(f"nnnnno inferred_pg")
        return ' '
    if(new_col_parts_name not in group[1]):
        print(f"nnnnno tz_inferred")
        return ' '    
    if(group[1][new_col_parts_name]==' '):
        print(f"nnnnno tz_inferred")
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

   
    adc_image =reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict['adc'][0], sitk.sitkBSpline)                                 
    hbv_image =reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict['hbv'][0], sitk.sitkBSpline)                                 
                                                  
    registered_prostate= reg_a_to_b_by_metadata_single_d(sources_dict[main_modality][0],sources_dict[new_col_name][0], sitk.sitkNearestNeighbor)



    # t2w_arr=sitk.GetArrayFromImage(sitk.ReadImage(group[1][main_modality][0]))
        
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
        labRes=labRes+(reduced_common.astype(int))

        # print(f"reduced_sum {np.sum(reduced_sum.flatten())}  reduced_common {np.sum(reduced_common.flatten())} labRes 2 {np.sum((labRes==2).flatten())}")







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
    
    inferred_parts_image= sitk.ReadImage(sources_dict[new_col_parts_name][0])

    inferred_parts_image_arr=sitk.GetArrayFromImage(inferred_parts_image)


    label_image = get_from_arr(labRes,t2w_image)

    pz_arrr=(inferred_parts_image_arr==1)
    tz_arrr=(inferred_parts_image_arr==2)
    

    pz_image = get_from_arr(pz_arrr.astype(int),t2w_image)
    tz_image = get_from_arr((inferred_parts_image_arr==2).astype(int),t2w_image)


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
    writer.SetFileName(out_pathsDict[new_col_parts_name])
    writer.Execute(pz_image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_pathsDict['tz_inferred'])
    writer.Execute(tz_image)

    return group[0]


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

sourceFrame=add_inferred_full_prost_to_dataframe(dir_inferred_prost, sourceFrame,new_col_name,out_folder)
sourceFrame=add_inferred_full_prost_to_dataframe(dir_inferred_prost, sourceFrame,new_col_parts_name,out_folder)
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
                                    ,pmap(partial(add_files_custom,main_modality=main_modality,modalities_of_intrest=modalities_of_intrest,non_mri_inputs=non_mri_inputs,labelsTrFolder=labelsTrFolder,imagesTrFolder=imagesTrFolder))                            
                                    ,list
                                    ,filter(lambda el: el!=' ')
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
   'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 12, 'patch_size': [32, 96, 96]
    #   'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2, 'patch_size': [96, 96, 96] # for swin
                                                , 'median_image_size_in_voxels': [32., 84., 95.]
, 'spacing': [0.78125, 0.78125   , 0.78125   ] #for swin
# , 'spacing': [3.30000019, 0.78125   , 0.78125   ]

, 'normalization_schemes': ['NoNormalization', 'NoNormalization', 'ZScoreNormalization', 'NoNormalization', 'NoNormalization']
, 'use_mask_for_norm': [False, False, False, False, False]
, 'UNet_class_name': 'PlainConvUNet'
, 'UNet_base_num_features': 32
, 'n_conv_per_stage_encoder': (2, 2, 2, 2, 2)
, 'n_conv_per_stage_decoder': (2, 2, 2, 2)
, 'num_pool_per_axis': [2, 4, 4]
, 'pool_op_kernel_sizes': [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
# , 'conv_kernel_sizes': [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
, 'conv_kernel_sizes': [[1, 5, 5], [1, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
, 'unet_max_num_features': 320
, 'resampling_fn_data': 'resample_data_or_seg_to_shape'
, 'resampling_fn_seg': 'resample_data_or_seg_to_shape'
, 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}
, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}
, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape'
, 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'batch_dice': False}




json_string = json.dumps(plans)     
print(f"aaaaaaaaaaaaa {json_string} \n \n \n ppppppppppppppppppppppppppppppppppp")
     
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


# cp -a /home/sliceruser/nnunetMainFolder/nnUNet_raw/Dataset101_Prostate/imagesTr /workspaces/konwersjaJsonData/explore/preprocessed


#### image_processing_oneformer is modified


# my_proj_name="seg lesions 4" tag="l4a test" my_proj_desc="l4a test" nnUNetv2_train 101 3d_lowres 0 -tr My_pl_trainer
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