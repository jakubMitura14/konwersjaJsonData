import SimpleITK as sitk
import itk
import mdai
import pandas as pd
import numpy as np
import cv2
import mainFuncs
import pydicom
import os
import multiprocessing as mp
import functools
from functools import partial
import mdai
import math
import time

JSON = '/workspaces/konwersjaJsonData/mdai_public_project_gaq3y0Rl_annotations_dataset_D_gQm1nQ_2022-07-15-104055.json'
results = mdai.common_utils.json_to_dataframe(JSON)
#so we have dictionary of dataframes
results.keys()#'annotations', 'studies', 'labels'
annot=results['annotations']

#path to folder with all required data
dataDir='/workspaces/konwersjaJsonData/data'
outputDir='/workspaces/konwersjaJsonData/output'
files_df= mainFuncs.get_df_file_info(dataDir)

def mainGenereteFiles(files_df,annot_for_series,currentSeries,studyPath):
    """
    main functions that gives single series of single annotator specialist
    will create files with 3d mha of this series plus 3d nii.gz files for each annotation label
    """
    ### 1) now we will get single series - get all paths of files related to it
    paths_in_series= files_df.loc[files_df['SeriesInstanceUID'] == currentSeries]['paths'].to_numpy() 
    ### 2) we will check how many diffrent labels are associated 
    uniq_labels= np.unique(annot_for_series['labelName'].to_numpy())

    copiedPath=os.path.join(studyPath,currentSeries )
    origVolPath = os.path.join(copiedPath ,'origVol')
    os.makedirs(origVolPath ,exist_ok = True)
    # into each subfolder we will copy the full  set of files related to main image at hand
    for path_to_copy in paths_in_series:
        os.system(f'cp {path_to_copy} {origVolPath}')  

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(origVolPath)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(origVolPath, series_IDs[0])

    #getseries file names in correct order
    #series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(copiedPath, currentSeries)
    series_reader = sitk.ImageSeriesReader()
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    series_reader.SetFileNames(series_file_names)

    image3D = series_reader.Execute()
    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    #writer.KeepOriginalImageUIDOn()
    newPath= os.path.join(copiedPath,'volume.mha')
    writer.SetFileName(newPath)
    writer.Execute(image3D)   
    data=sitk.GetArrayFromImage(image3D)

    ### we will change the data type of each file into boolean and fil it with zeros
    #first get all files paths - but not of the original file
    for lab in uniq_labels:
        dtype=np.uint16
        annot_for_label=annot_for_series.loc[annot_for_series['labelName'] == lab]
        
        zeroArray=np.zeros(data.shape, dtype=dtype)
        print(f"data.shape {data.shape}")
        for index,loccPath in enumerate(series_file_names):
            #open file to get sop id
            ds = pydicom.dcmread(loccPath)
            sop=mainFuncs.get_SOPInstanceUID(ds)
            annot_for_sop= annot_for_label.loc[annot_for_label['SOPInstanceUID'] == sop]       
            if(len(annot_for_sop)>0):
                print("overWriting")
                rowOfIntr=list(annot_for_sop.iterrows())[0]
                #we obtain mask as a boolean array
                binArray= mainFuncs.load_mask_instance(rowOfIntr).astype(dtype)  
                print(f"binArray {binArray} lab {lab}")
                #time to overwrite the data
                # ds.PixelData = binArray.tostring() 
                zeroArray[index,:,:]=binArray

        # data is already overwritten
        # reading series of dicom and save them as nii.gz in case of the 
        # from https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html
        #series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(locPath)
        image = sitk.GetImageFromArray(zeroArray)  
        image.SetSpacing(image3D.GetSpacing())
        image.SetOrigin(image3D.GetOrigin())
        image.SetDirection(image3D.GetDirection())    

        newPath= os.path.join(copiedPath,lab+'.nii.gz')
        writer.SetFileName(newPath)
        writer.Execute(image)   


## get single study
for current_study_id in np.unique(annot['StudyInstanceUID'].to_numpy()):
    annot_for_study_id=annot.loc[annot['StudyInstanceUID'] == current_study_id]
    #get annotator id 
    for current_doctor_id in np.unique(annot_for_study_id['createdById'].to_numpy()):
        annot_for_doctor=annot_for_study_id.loc[annot_for_study_id['createdById'] == current_doctor_id]
        #create directory for this study
        studyPath = os.path.join(outputDir, current_study_id,current_doctor_id)
        os.makedirs(studyPath, exist_ok = True)
        #get single series
        for currentSeries in np.unique(annot_for_doctor['SeriesInstanceUID'].to_numpy()):
            annot_for_series=annot_for_doctor.loc[annot_for_doctor['SeriesInstanceUID'] == currentSeries]
            mainGenereteFiles(files_df,annot_for_series,currentSeries,studyPath)