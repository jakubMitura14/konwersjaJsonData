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
from os.path import basename, dirname, exists, isdir, join, split
import shutil
from mainFuncs import getLabelsAbbrev
import os
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape,shape2D
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
import SimpleITK as sitk
from radiomics import featureextractor, getTestCase
import six 
from napari_simpleitk_image_processing import label_statistics

settings = {'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None}



def groupByMaster(rowws):
    grouped_by_master= groupby(lambda row : row[1]['masterolds'],rowws)
    # grouped_by_master=[(key,list(group)) for key, group in grouped_by_master]
    return dict(grouped_by_master).items()

def get_2d_stats(image_arr, mask_arr,new_spacing):
    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)
    image.SetSpacing(new_spacing)
    mask.SetSpacing(new_spacing)

    # statistics = label_statistics(image, mask, 
    #                             shape=True, 
    #                             perimeter=True, 1
    #                             position=True,
    #                             moments=True)

    # df = pd.DataFrame(statistics)
    # print(df)
    # shapeFeatures = shape2D.RadiomicsShape2D(mask, mask)#,force2D=True
    # # shapeFeatures = shape2D.RadiomicsShape2D(image, mask)#,force2D=True

    # shapeFeatures.enableAllFeatures()
    # results = shapeFeatures.execute()
    # return results['Perimeter']


    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    perimeters_of_labels=np.array([ stats.GetPerimeter(l) for l in stats.GetLabels() ]) #if l != 1
    if(len(perimeters_of_labels)==0):
        return 0.0
    return np.max(perimeters_of_labels)





def get_perimeter(t2w_path, change_path):
    """ 
    using getPerimeterFeatureValue from pyradiomics to calculate the perimeter of the change in the image
    """
    image = sitk.ReadImage(t2w_path)
    mask = sitk.ReadImage(change_path)

    spacing=mask.GetSpacing()
    new_spacing= (spacing[0],spacing[1]) 
    
    
    three_d_arr_im = sitk.GetArrayFromImage(image)
    three_d_arr_mask = sitk.GetArrayFromImage(mask)

    shhh= three_d_arr_mask.shape
    z_dim= shhh[0]
    perimeters= list(map(lambda z: get_2d_stats(three_d_arr_im[z,:,:], three_d_arr_mask[z,:,:],new_spacing) , range(z_dim) ))
    return np.max(np.array(perimeters).flatten())



def getPathsFromRows(colName,listRows):
    """
    extracting all paths of intrest from row
    """
    res=  list(map( lambda row : row[1][colName] ,listRows ))
    return list(filter(lambda el: el!=' ',res ))


def iterGroups(groupTuple,anatomy_cols ):
    masterOlds, listRows= groupTuple
    #first find all of the lesion paths and anatomy paths
    anatomy_paths=toolz.pipe(anatomy_cols
                ,map(partial(getPathsFromRows,listRows=listRows))
                ,list) 

    anatomy_paths=list(filter(lambda  el: len(el)>0 , anatomy_paths))
    mri_paths = list(map(lambda row: (row[1]['series_desc'],row[1]['series_MRI_path']) ,listRows))
    t2w = list(filter(lambda el: el[0]== 't2w' ,mri_paths ))[0]
    # now we need a path to t2w and adc
    return (masterOlds,t2w[1],list(zip(anatomy_cols,anatomy_paths)))


def get_circumFerences(groupTuple):
    masterOlds,t2w,anatomy_paths=groupTuple
    anatomy_names,anatomy_paths_pure = list(toolz.sandbox.core.unzip( anatomy_paths ))
    anatomy_names=list(anatomy_names)
    anatomy_paths_pure=list(anatomy_paths_pure)
    max_parimeters=list(map(lambda change_path: get_perimeter(t2w, change_path),np.array(anatomy_paths_pure).flatten()))
    return (masterOlds,dict(list(zip(anatomy_names, max_parimeters))))
    


def save_max_paerimeter(sourceFrame,anatomy_cols,circ_frame_csv_dir):
    circums=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
    #with mp.Pool(processes = 1) as pool:
        @curry  
        def pmap(fun,iterable):
            return pool.map(fun,iterable)


        circums= toolz.pipe(sourceFrame.iterrows()
                                ,groupByMaster
                                ,pmap(partial(iterGroups,anatomy_cols=anatomy_cols ))
                                # ,filter(lambda group: ' ' not in group[1].keys() )
                                ,list
                                ,pmap(get_circumFerences)
                                ,list   
                                ,filter(lambda el: el!=' ') 
                                ,list                                                        )

    circ_frame= pd.DataFrame()
    #populating frame with data
    circ_frame['id']=list(map(lambda el : el[0]  ,circums))
    
    for col_name in anatomy_cols:
        circ_frame[col_name]=list(map(lambda el : el[1].get(col_name,' ')  ,circums))
    circ_frame.to_csv(circ_frame_csv_dir) 

    return circ_frame

# (masterOlds,dict(list(zip(anatomy_names, mean_adcs))))
