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
import nnunet
import nnunet.dataset_conversion
from nnunet.dataset_conversion.utils import generate_dataset_json
import elastixRegister
from elastixRegister import reg_a_to_b
import tempfile
import shutil
import re
from toolz.itertoolz import groupby
from toolz import curry

resCSVDir='/workspaces/konwersjaJsonData/outCsv/resCSV.csv'

elacticPath='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/elastix'
transformix_path='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/transformix'
reg_prop='/home/sliceruser/registration/parameters.txt'  


nNunetBaseFolder='/workspaces/konwersjaJsonData/nnunetMainFolder'
taskName= 'Task505_opiProstate'
taskFolder = join(nNunetBaseFolder,'nnUNet_raw_data_base','nnUNet_raw_data',taskName)
imagesTrFolder= join(taskFolder,'imagesTr')
labelsTrFolder= join(taskFolder,'labelsTr')
imagesTsFolder= join(taskFolder,'imagesTs')
# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']
# main modality that will be set as a reference and others will be registered to it 
main_modality = 't2w'

def groupByMaster(rowws):
    grouped_by_master= groupby(lambda row : row[1]['masterolds'],rowws)
    # grouped_by_master=[(key,list(group)) for key, group in grouped_by_master]
    return dict(grouped_by_master).items()

os.makedirs(join(nNunetBaseFolder,'nnUNet_raw_data_base') ,exist_ok = True)
os.makedirs(join(nNunetBaseFolder,'nnUNet_raw_data_base','nnUNet_raw_data') ,exist_ok = True)
os.makedirs(taskFolder ,exist_ok = True)
os.makedirs(imagesTrFolder ,exist_ok = True)
os.makedirs(labelsTrFolder ,exist_ok = True)
os.makedirs(imagesTsFolder ,exist_ok = True)





sourceFrame = pd.read_csv(resCSVDir) 
cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))+['series_MRI_path']

lesion_cols=list(filter(lambda el: 'lesion' in el , noSegCols))


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
    getting patsh related to single modality and extracting main MRI image
    """
    mod_paths = list(filter(lambda pathh :modalityName in  pathh,pathhs))
    mri = list(filter(lambda el: '.mha' in el ,mod_paths))
    if(len(mri)==0):
        return ' ',[]
    mri=mri[0]   
    mod_paths= list(filter(lambda pathh: pathh!=mri , mod_paths))

    return (modalityName,(mri,mod_paths))

def myFlatten(liist):
    return  list(itertools.chain(*liist))
def map_modalities(pathhs,modalities):
    return map(partial(getListModality,pathhs=pathhs),modalities)

def iterGroupModalities(groupTuple,modalities_of_intrest ):
    """
    grouping the paths into dictionary relative to modalities they represent and lesions on thise 
    modalities
    """
    masterOlds, listRows= groupTuple
    pathhs=toolz.pipe(listRows
                ,map(partial(getPathsFromRow,list_columns=lesion_cols+['series_MRI_path']))
                ,myFlatten
                ,filter(lambda el : len(el)>2)
                ,list
                ,partial(map_modalities,modalities=modalities_of_intrest)
                ,dict
                )   
    return (masterOlds,pathhs)

#currently we will filter out just adc hbv and t2w
modalitiesOfIntrest = ['t2w','adc','hbv' ]

grouped_rows= toolz.pipe(sourceFrame.iterrows()
                        ,filter(lambda row: row[1]['series_desc'] in modalitiesOfIntrest)
                        ,groupByMaster
                        ,map(partial(iterGroupModalities,modalities_of_intrest=modalities_of_intrest))
                        # ,map(lambda el: el[1].keys() )
                        #  ,filter(lambda group: ' ' not in group[1].keys() )
                        ,list

  )




group = grouped_rows[1]
print(group)



#     if(pathhs_t2w[0]==' ' or pathhs_adc[0]==' ' or pathhs_hbv[0]==' ' ):
#         return (' ', {' ': []})




# temp_dir = tempfile.mkdtemp() #
# temp_dir = '/workspaces/konwersjaJsonData/explorationn'

# reg_adc,adc_labs=reg_a_to_b(temp_dir,group[0],group[1]['t2w'][0],group[1]['adc'][0],group[1]['adc'][1],reg_prop ,elacticPath,transformix_path)



# # modalities_of_intrest=['t2w','adc','hbv']
# # # main modality that will be set as a reference and others will be registered to it 
# # main_modality = 't2w'



# toolz.pipe(adc_labs, 
#     map(get_bool_arr_from_path)
#     ,functools.reduce(np.logical_or)
#     ,list
# )

# # ... do stuff with dirpath
# shutil.rmtree(temp_dir)








#  All images (including labels) must be 3D nifti files (.nii.gz)!

#  example image Name

# t2w opiProstate_{master with zeros}_0000.nii.gz
# adc opiProstate_{master with zeros}_0001.nii.gz

# label
# adc opiProstate_{master with zeros}.nii.gz

