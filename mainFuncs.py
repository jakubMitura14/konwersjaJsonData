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
import math
from os.path import basename, dirname, exists, isdir, join, split
from os import path as pathOs

def load_mask_instance(row):
    """Load instance masks for the given annotation row. Masks can be different types,
    mask is a binary true/false map of the same size as the image.
    """
    row=row[1]
    if(not math.isnan(row.height) and not math.isnan(row.width) and isinstance(row["data"],dict)  ):
        mask = np.zeros((int(row.height), int(row.width)), dtype=np.uint8)

        annotation_mode = row.annotationMode
        # print(annotation_mode)

        if annotation_mode == "bbox":
            # Bounding Box
            x = int(row["data"]["x"])
            y = int(row["data"]["y"])
            w = int(row["data"]["width"])
            h = int(row["data"]["height"])
            mask_instance = mask[:,:].copy()
            cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
            mask[:,:] = mask_instance


        # FreeForm or Polygon
        elif annotation_mode == "freeform" or annotation_mode == "polygon":
            vertices = np.array(row["data"]["vertices"])
            vertices = vertices.reshape((-1, 2))
            mask_instance = mask[:,:].copy()
            cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
            mask[:,:] = mask_instance

        # Line
        elif annotation_mode == "line":
            vertices = np.array(row["data"]["vertices"])
            vertices = vertices.reshape((-1, 2))
            mask_instance = mask[:,:].copy()
            cv2.polylines(mask_instance, np.int32([vertices]), False, (255, 255, 255), 12)
            mask[:,:] = mask_instance

        elif annotation_mode == "location":
            # Bounding Box
            x = int(row["data"]["x"])
            y = int(row["data"]["y"])
            mask_instance = mask[:,:].copy()
            cv2.circle(mask_instance, (x, y), 7, (255, 255, 255), -1)
            mask[:,:] = mask_instance

        elif annotation_mode == "mask":
            mask_instance = mask[:, :].copy()
            if a.data["foreground"]:
                for i in a.data["foreground"]:
                    mask_instance = cv2.fillPoly(mask_instance, [np.array(i, dtype=np.int32)], (255, 255, 255))
            if a.data["background"]:
                for i in a.data["background"]:
                    mask_instance = cv2.fillPoly(mask_instance, [np.array(i, dtype=np.int32)], (0,0,0))
            mask[:, :] = mask_instance

        elif annotation_mode is None:
            print("Not a local instance")


        return mask.astype(np.bool)
    return np.full((2, 2,2), False)


# convinience functions to load dicom metadaa from pydicom object 

def get_SOPInstanceUID(ds):
    return ds.file_meta[(0x0002, 0x0003)].repval

def get_patient_id(ds):
    return ds[(0x0010, 0x0020)].repval

def get_SeriesInstanceUID(ds):
    return ds[(0x0020, 0x000e)].repval

def get_StudyInstanceUID(ds):
    return ds[(0x0020, 0x000d)].repval

def get_all_file_paths(dataDir):
    """
    get ll files with their full paths in a given folder and subfolders
    """
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dataDir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles    


def getUids(row):
    """
    given path from row of data frame will return information about SOP series and study ids 
    """
    row=row[1]
    path=str(row['paths'])
    try:
        ds = pydicom.dcmread(path)
        sop=get_SOPInstanceUID(ds)
        SeriesInstanceUID= get_SeriesInstanceUID(ds)
        InstanceUID=get_StudyInstanceUID(ds)
        return(sop,SeriesInstanceUID,InstanceUID  )
    except:
        return ('','','')   


def get_df_file_info(dataDir,client_down_csv):
    """
    get paths and UIds related to given directory if the rtelated csv is not yet present
    """
    if(pathOs.exists(client_down_csv)):
        return pd.read_csv(client_down_csv)
    listOfPatsh=get_all_file_paths(dataDir)
    df = pd.DataFrame()
    df['paths']=listOfPatsh
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(getUids  ,list(df.iterrows()))

    sops = list(map(lambda tupl: tupl[0], resList))
    SeriesInstanceUIDs = list(map(lambda tupl: tupl[1], resList))
    InstanceUIDs = list(map(lambda tupl: tupl[2], resList))
    df['SOPInstanceUID']=sops   
    df['SeriesInstanceUID']=SeriesInstanceUIDs   
    df['StudyInstanceUID']=InstanceUIDs
    df.to_csv(client_down_csv)       
    return df        