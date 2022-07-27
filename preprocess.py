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

### first one need to invoke the get3dFiles to produce dataframe and save it in resCSVDir
resCSVDir='/workspaces/konwersjaJsonData/resCSV'

JSON = '/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_dataset_D_gQm1nQ_2022-07-15-104055.json'
results = mdai.common_utils.json_to_dataframe(JSON)
#so we have dictionary of dataframes
results.keys()#'annotations', 'studies', 'labels'
annot=results['annotations']

filesPandas = pd.read_csv(resCSVDir)

#first filter out only the rows with prostate as only this is intresting for us at the moment 
all_labels_types=np.unique(annot['labelName'].to_numpy())
all_labels_types

labelsOfIntrest = ['peripheral zone', 'prostate', 'transition zone', 'urethra']



eqq =torch.logical_and(y_true,y_hat)  #torch.eq(labelBoolTensorA,summB)
#false negatives labelBoolTensorA - gold stadard
fn=  torch.logical_and(torch.logical_not(eqq),y_true)
#false positives labelBoolTensorA - gold stadard
fp=  torch.logical_and(torch.logical_not(eqq),y_hat)
####first we will look around of all fp points     
num_points_fp = torch.sum(fp).item()
#to sparse - to get indicies - that transpose for correct dim
fpIndicies = torch.t(fp.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 



    goldIndicies =  torch.t(y_true.to_sparse().indices()).type(torch.float32).contiguous().to('cuda') 





