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
import re
from os.path import basename, dirname, exists, isdir, join, split
import shutil
import toolz
from toolz.curried import pipe, map, filter, get
from toolz import curry

def getListMeta(currentDicomDirPath):
    """
    iterates ove files associated with single dicom dir
    and returns the metadata related to them
    """
    pathhh = Path(currentDicomDirPath)
    currentFold=str(os.path.relpath(pathhh.parent))
    ds = pydicom.dcmread(currentDicomDirPath)
    fs = FileSet(ds)
    fileMataDatas=list(map(partial(getFileData,currentFold=currentFold),list(fs)))
    return fileMataDatas

def getFileData(current_file_ref,currentFold):
    """
    given file return its series, sop and study UID
    """
    filePath=current_file_ref.path
    current_file = current_file_ref.load()
    sop=mainFuncs.get_SOPInstanceUID(current_file)
    SeriesInstanceUID= mainFuncs.get_SeriesInstanceUID(current_file)
    stidy_instanceUID=mainFuncs.get_StudyInstanceUID(current_file)
    return (currentFold,filePath,stidy_instanceUID,SeriesInstanceUID,sop  )


def get_df_orig_dir_info(orig_data_dir,csvDir):
    """
    iterates over files and saves ther paths and matadata to dataframe and subsequently to csv
    """
    if(pathOs.exists(csvDir)):
        return pd.read_csv(csvDir)    
    #get all paths of DICOMDIR files
    listOfPatsh=mainFuncs.get_all_file_paths(orig_data_dir)
    pathsDICOMDIR= list(filter(lambda pathh:  'DICOMDIR' in pathh,listOfPatsh ))
    df = pd.DataFrame()
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(getListMeta ,pathsDICOMDIR)
    #flatten array
    resList=list(itertools.chain(*resList))

    masterolds=list(map(lambda tupl: tupl[0], resList))
    currentFilePath=list(map(lambda tupl: tupl[1], resList))
    StudyInstanceUID = list(map(lambda tupl: tupl[2], resList))
    SeriesInstanceUIDs = list(map(lambda tupl: tupl[3], resList))
    sops = list(map(lambda tupl: tupl[4], resList))

    df['masterolds']=masterolds   
    df['paths']=currentFilePath   
    df['SOPInstanceUID']=sops   
    df['SeriesInstanceUID']=SeriesInstanceUIDs   
    df['StudyInstanceUID']=StudyInstanceUID  
    df.to_csv(csvDir) 
    return df 

def get_orig_fold(uid,dictt):
    """
    getting the folder name in original folder
    """
    if(uid in dictt):
            return dictt[uid]
    return " "       

def add_orig_dir_data(files_df, files_df_origFolds):
       """
       we got the data about original folder s from orig folders frame
       now we add this data to data downloaded from mdai by client library
       """
       smaller_df=files_df_origFolds[['masterolds','StudyInstanceUID']].drop_duplicates()
       masterolds=smaller_df['masterolds'].to_numpy()
       StudyInstanceUID=smaller_df['StudyInstanceUID'].to_numpy()
       dictt ={StudyInstanceUID[i]: masterolds[i] for i in range(len(masterolds))}
       StudyInstanceUID_mdai= files_df['StudyInstanceUID'].to_numpy()
       orig_folds_dat= list(map(lambda uid :get_orig_fold(uid,dictt) ,StudyInstanceUID_mdai))
       files_df['masterolds']=orig_folds_dat
       return files_df






    # from ipyparallel import Client
    # p = Client()[:]
    # pmap = p.map_sync

    # total = merge_with(sum, pmap(wordcount, filenames))



@curry
def get_fromTag(tag,row):
    if(row['SeriesInstanceUID']==' '):
        return ' '
    ds = pydicom.dcmread(row['paths'])
    res=' '
    try:
        res=ds[tag].repval
    except:
        pass
        #print(f"no tag  {tag}")    
    return res

def mapTag(rowws,tag):
    return pipe(rowws
    ,map(lambda row : row[1])
    ,filter(lambda row :row['SeriesInstanceUID']!=' ')
    ,map(get_fromTag(tag))
    ,toolz.itertoolz.unique
    ,list)


def mapTagwithPath(rowws,tag):
    return pipe(rowws
    ,map(lambda row : row[1])
    ,filter(lambda row :row['SeriesInstanceUID']!=' ')
    ,map(lambda row : (row['paths'],get_fromTag(tag)(row) ))
    ,filter(lambda it : it[0]!=' ' )
    ,list)

# rowws=list(out_files_frame.iterrows())
# rowws=list(map(lambda row: row[1]  ,rowws))
# uniqseriesDesc_with_path=list(toolz.itertoolz.unique(rowws,key=lambda row : row['series_desc']))
# list(map(lambda row : row['series_desc'],uniqseriesDesc_with_path))

# uniqseriesDesc_with_path
# len(uniqseriesDesc_with_path)

# row=uniqseriesDesc_with_path[1]
# strr=row['series_desc'].replace("'","")+'.mha'
# join(uniqseriesDesc_with_path, +'.mha')

# pathExploreFolder='/workspaces/konwersjaJsonData/exploration'
# list(map(lambda row :
#     shutil.copyfile(row['series_MRI_path'], join(pathExploreFolder, row['series_desc'].replace("'","")+'.mha')) 
#      ,uniqseriesDesc_with_path))

# seriesTag=(0x0008, 0x103e)
# modalityTag=(0x0008, 0x0060)
# scanningSeqTag=(0x0018, 0x0020)
# SequenceNameTag=(0x0018, 0x0024)
# angioFlagTag=(0x0018, 0x0025)
# planeTag= (0x0051, 0x100e)

# rowws=list(files_df.iterrows())
# shutil.copyfile(mriPath, join(locFolderPath, 'volume.mha'))

# uniqseriesDesc=mapTag(rowws,seriesTag)
# seriesDesc_with_path=mapTagwithPath(rowws,seriesTag)

# uniqseriesDesc_with_path=list(toolz.itertoolz.unique(seriesDesc_with_path,key=lambda tupl: tupl[1]))

# len(uniqseriesDesc_with_path)

# len(uniqseriesDesc)
# uniqseriesDesc_with_path[2]

# pathExploreFolder='/workspaces/konwersjaJsonData/exploration'
# list(map(lambda tupl :
#     shutil.copyfile(tupl[0], join(pathExploreFolder, tupl[1]+'.mha')) 
#      ,uniqseriesDesc_with_path))


# pathsForDesc=pipe(uniqseriesDesc
# ,map(lambda descc : list(filter(lambda row : get_fromTag(row[1])==descc ,rowws ))[0]   )
# # ,map(lambda row ,row[1]['paths'] )
# ,list
# )

# uniqModalityTag= mapTag(rowws,modalityTag)
# uniqscanningSeqTag= mapTag(rowws,scanningSeqTag)
# uniqSequenceNameTag= mapTag(rowws,SequenceNameTag)
# uniqangioFlagTag= mapTag(rowws,angioFlagTag)
# uniqPlaneTag= mapTag(rowws,planeTag)

# uniqseriesDesc
# uniqModalityTag
# uniqscanningSeqTag
# uniqSequenceNameTag
# uniqangioFlagTag
# uniqPlaneTag
"""
https://www.mr-tip.com/serv1.php?type=cam
Siemens dicom conformance statement
https://marketing.webassets.siemens-healthineers.com/1800000001958391/135cf501a2b8/conformance_dc_ve11-01958391_1800000001958391.pdf


adc - "'ep2d_diff_b 50 400 800 1200_ADC'"
t2 transverse - "'t2_bl_tse_fs_tra'", "'t2_bl_tse_tra_P'"
    ,"'t2_bl_tse_tra_p'","'t2_bl_tse_fs_tra'","'t2_bl_tse_tra'",t2_bl_tse_tra_p'"

>>> uniqseriesDesc
["'t2_bl_tse_sag'", "'t2_bl_tse_tra_P'"
, "'t2_bl_tse_cor'", "'ep2d_diff_b 50 400 800 1200_TRACEW'"
, "'ep2d_diff_b 50 400 800 1200_ADC'", "'t1_fl3d_tra fs_dyn CM'"
, "'t1_fl3d_tra fs_dyn CM_PEI'", "'t2_bl_tse_tra_p'", ' '
, "'t2_trufi_cor_loc multi'", "'t1_fl3d_tra fs_dyn CM_SUB_MOCO'"
, "'WASH-IN'", "'WASH-OUT'", "'TTP'", "'iAUC'", "'AT'", "'PEI'"
, "'t1_fl3d_tra fs_dyn CM_SUB'", "'t1_tse_sag_fs_CM'"
, "'t1_tse_tra_CM'", "'t1_vibe_dix_cor_bh_opp'"
, "'t1_vibe_dix_cor_bh_in'", "'t1_vibe_dix_cor_bh_W'"
, "'t1_fl2d_tra_mbh_CM'", "'t1_tse_tra'"
, "'t2_bl_tse_fs_tra'", "'t1_fl3d_tra fs_dyn CM_SUB_MIP_COR'"
, "'t2_bl_tse_tra'", "'t2_bl_tse_COR_P'", "'t2_bl_SAG_tra_P'"
, "'t1_tse_sag'", "'t1_tse_tra_fs_CM'", "'t1_fl3d_cor_Dix_CM_opp'"
, "'t1_fl3d_cor_Dix_CM_in'", "'t1_fl3d_cor_Dix_CM_W'"
, "'t1_tse_cor limf nodes CM'", "'[KOELIS] t2_bl_tse_tra_p'"
, "'t2_bl_tse_tra_sFOV'", "'t1_fl3d_tra_dyn CM'"
, "'t1_fl3d_tra_dyn CM_PEI'"]
>>> uniqModalityTag
["'MR'"]
>>> uniqscanningSeqTag
["'SE'", "'EP'", "'GR'", ' ']
>>> uniqSequenceNameTag
["'*tseBR2d1_43'", "'*tseBR2d1_32'", "'*ep_b1200t'", "'*ep_b50_1200'", "'*fl3d1'", "'*tseB2d1_29'", "'*ep_b50t'", "'*ep_b400t'", "'*ep_b800t'", "'*tfi2d1_134'", ' ', "'*tse2d1_4'", "'*fl3d2'", "'*fl2d1'", "'*tseB2d1_35'", "'*tseBRW2d1_43'", "'*tseBW2d1_29'", "'*tse2d1_3'", "'*ep_b450t'"]
>>> uniqangioFlagTag
["'N'", ' ']
>>> 
"""


# row=rowws[10][1]
# row
# ds = pydicom.dcmread(row['paths'])
# ds



# seriesDs = pipe(rowws
# ,map(lambda row : row[1])
# ,filter(lambda row :row['SeriesInstanceUID']!=' ')
# ,map(get_modality)
# ,toolz.itertoolz.unique
# ,list)
# seriesDs
# res

# (0008, 0060) 


# get_SeriesDesc(res[9])



# series_desc= list(map(  ,rowws))


# (0008, 103e) Series Description                  LO: 't2_bl_tse_sag'