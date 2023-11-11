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
from nnunet.dataset_conversion import generate_dataset_json
import shutil
import zipfile
import glob

resCSVDir='//home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
mainFoldDirMha='//home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont'
mainFoldDirSeg='//home/sliceruser/workspaces/konwersjaJsonData/AI4AR_dicom'


mainFoldDirSegChunked='//home/sliceruser/workspaces/konwersjaJsonData/AI4AR_dicom_chunked'
mainFoldDirMhaChunked='//home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont_chunked'


sourceFrame = pd.read_csv(resCSVDir) 
cols=sourceFrame.columns
noSegCols=list(filter(lambda el: '_noSeg' in el , cols))
segCols=list(filter(lambda el: '_Seg' in el , cols))+['dicom_MRI_path']

masterOlds=np.unique(sourceFrame["masterolds"].to_numpy())
chunked =np.array_split(masterOlds, 16)



def getParentPath(pathh):
    return Path(pathh).parent
def getEntries(row, colss):
    return list(map(lambda colName: row[colName],colss))



def copyTreeee(sourceFrame,masterNum, colNames,oldDir,chunkedDir,index ):
    locFrame = sourceFrame.loc[sourceFrame['masterolds'] == masterNum]
    rowws = list(locFrame.iterrows())
    rowws= list(map(lambda el: el[1] ,rowws))

    parentss = list(map(lambda row : getEntries(row, colNames) ,rowws ))
    parentss= list(itertools.chain(*parentss))
    parentss= list(filter(lambda el: el!=' ',parentss))
    parentss= list(map(getParentPath,parentss))
    parentss= np.unique(parentss)
    parentssNew= list(map(lambda path: path.as_posix().replace(oldDir,join(chunkedDir,str(index)) ),parentss))
    parentssNew= list(map(lambda path:Path(path) ,parentssNew))
    zipped = list(zip(parentss,parentssNew))

    for tupl in zipped:
        shutil.copytree(tupl[0], tupl[1], dirs_exist_ok=True)




for tupl in enumerate(chunked):
    index,chunk= tupl
    for masterNum in chunk:
        copyTreeee(sourceFrame,masterNum, segCols,mainFoldDirSeg,mainFoldDirSegChunked,index )
        copyTreeee(sourceFrame,masterNum, noSegCols,mainFoldDirMha,mainFoldDirMhaChunked,index )

for tupl in enumerate(chunked):
    index,chunk= tupl
    for masterNum in chunk:
        with zipfile.ZipFile(fp_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fp in glob(os.path.join(parent, "**/*")):
                base = os.path.commonpath([parent, fp])
                zipf.write(fp, arcname=fp.replace(base, ""))


        # locFrame = sourceFrame.loc[sourceFrame['masterolds'] == masterNum]
        # rowws = list(locFrame.iterrows())
        # rowws= list(map(lambda el: el[1] ,rowws))
        
        # for row in rowws:
        #     for segCol in segCols:
        #         currPath=row[segCol]
        #         newPath = currPath.replace(mainFoldDirSeg,join(mainFoldDirSeg,str(index)))
        #         shutil.copy(source, destination)
        #         shutil.copytree('baz', 'foo', dirs_exist_ok=True)

chunked[0]

