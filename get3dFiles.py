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

from os import path as pathOs

def getSliceAndSOP(filename):
    """
    finds SOP uid of the given dicom file
    """
    ds = pydicom.dcmread(filename)
    return (mainFuncs.get_SOPInstanceUID(ds),ds.pixel_array ,filename)

def findTheSame(slice,slices_with_sop,sops_in_anot):
    """
    used in order to be sure that we do not have two slices with the same data
    """
    # we look for all slices holding the same data
    theSame= list(filter(lambda tupl : np.array_equal(tupl[1],slice)  ,slices_with_sop))
    #in case we have just single slice data with such data we will return just this single slice
    first=theSame[0]
    if len(theSame)==1:
        return first
    #if we have multiple we need to prefer this that is included in some annotation     
    for tupll in theSame:
        if(tupll[0] in sops_in_anot):
            return tupll
    #in case it is not in annotation we return first from array (any is the same)
    return first         


def createLabelFile(annot_for_series,lab,data,copiedPath,series_file_names,image3D):
    """
    takes label dataframe and original MRI image and recreates label volume from it 
    """
    dtype=np.uint16
    annot_for_label=annot_for_series.loc[annot_for_series['labelName'] == lab]
    zeroArray=np.zeros(data.shape, dtype=dtype)

    newPathLab= os.path.join(copiedPath,lab+'.nii.gz')
    writer = sitk.ImageFileWriter()
    if(not pathOs.exists(newPathLab)):
        for index,loccPath in enumerate(series_file_names):
            #open file to get sop id
            ds = pydicom.dcmread(loccPath)
            sop=mainFuncs.get_SOPInstanceUID(ds)
            annot_for_sop= annot_for_label.loc[annot_for_label['SOPInstanceUID'] == sop]       
            if(len(annot_for_sop)>0):
                #print("overWriting")
                rowOfIntr=list(annot_for_sop.iterrows())[0]
                #we obtain mask as a boolean array
                binArray= mainFuncs.load_mask_instance(rowOfIntr).astype(dtype)  
                #time to overwrite the data
                # ds.PixelData = binArray.tostring()
                if((data.shape[1],data.shape[2])==binArray.shape):
                    #print("shapes ok lab {lab}")
                    zeroArray[index,:,:]=binArray
                else:
                    print(f"dimensions do not match lab {lab}")
                    print(f"data.shape {data.shape}")
                    print(f"binArray shape {binArray.shape} ")
                    print(f"binArray    {binArray} ")
    

        # data is already overwritten
        # reading series of dicom and save them as nii.gz in case of the 
        # from https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html
        #series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(locPath)
        image = sitk.GetImageFromArray(zeroArray)  
        image.SetSpacing(image3D.GetSpacing())
        image.SetOrigin(image3D.GetOrigin())
        image.SetDirection(image3D.GetDirection())    
    
        writer.SetFileName(newPathLab)
        writer.Execute(image)   
    return (lab,newPathLab)


def mainGenereteFiles(files_df,annot_for_series,currentSeries,studyPath,current_study_id,current_doctor_id):
    """
    main functions that gives single series of single annotator specialist
    will create files with 3d mha of this series plus 3d nii.gz files for each annotation label
    return current_study_id current_doctor_id series number, path to the main volume and list of tuples containing label name and 
    """
    ### 1) now we will get single series - get all paths of files related to it
    locDf = files_df.loc[files_df['SeriesInstanceUID'] == currentSeries]
    locDf=locDf.drop_duplicates(subset = ["SOPInstanceUID"])

    paths_in_series= np.unique(locDf['paths'].to_numpy())
    ### 2) we will check how many diffrent labels are associated 
    uniq_labels= np.unique(annot_for_series['labelName'].to_numpy())

    copiedPath=os.path.join(studyPath,currentSeries )
    origVolPath = os.path.join(copiedPath ,'origVol')


    # checking weather there are more than one file with the same data
    slices_with_sop = list(map(getSliceAndSOP ,paths_in_series))

    sops_in_anot= np.unique(annot_for_series['SOPInstanceUID'].to_numpy())
    
    filtered = list(map(lambda tupl :  findTheSame(tupl[1],slices_with_sop,sops_in_anot) ,slices_with_sop))
    paths_in_series= np.unique(list(map(lambda tupl: tupl[2],filtered)))


    newPath= os.path.join(copiedPath,'volume.mha')
    series_file_names=None
    image3D=None
    #avoiding creating file if one is present
    if(not pathOs.exists(newPath)): 

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
        writer.SetFileName(newPath)
        writer.Execute(image3D)   
        print(f"newPath image3D {newPath}")
    else:
        image3D=sitk.ReadImage(newPath)
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(origVolPath)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(origVolPath, series_IDs[0])

    data=sitk.GetArrayFromImage(image3D)


    labelNameAndPaths=list(map(lambda lab: createLabelFile(annot_for_series,lab,data,copiedPath,series_file_names,image3D),uniq_labels ))
    return (current_study_id,current_doctor_id,currentSeries,newPath,labelNameAndPaths  )

        


def iterate_overStudy(current_study_id,files_df,annot,outputDir):
    """
    iterate ove all series with the same study UID
    """
    res=[]
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
            res.append(mainGenereteFiles(files_df,annot_for_series,currentSeries,studyPath,current_study_id,current_doctor_id))
    return res



def getLabelPathOrEmpty(targetLab, tupl):
    """
    return path to the label file if it exists otherwise " "
    """
    listLabs=tupl[4]
    for labb in listLabs :
        if(labb[0]== targetLab):
            return labb[1]
    return " "    

def get_frame_with_output(files_df,annot,outputDir,resCSVDir):
    """
    in parallel iterates over all studies and series and save the paths of created files in the csv file
    """
    if(pathOs.exists(resCSVDir)):
        return pd.read_csv(resCSVDir) 
    
    out_files_frame= pd.DataFrame()
    #iterate over all files
    allPaths=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        allPaths=pool.map(partial(iterate_overStudy, files_df=files_df,annot=annot,outputDir=outputDir), np.unique(annot['StudyInstanceUID'].to_numpy()))


    # (current_study_id,current_doctor_id,currentSeries,newPath,labelNameAndPaths  )
    flatten_list_paths = list(itertools.chain(*allPaths))
    out_files_frame['study_id']=list(map(lambda tupl: tupl[0],flatten_list_paths))
    out_files_frame['doctor_id']=list(map(lambda tupl: tupl[1],flatten_list_paths))
    out_files_frame['series_id']=list(map(lambda tupl: tupl[2],flatten_list_paths))
    out_files_frame['series_MRI_path']=list(map(lambda tupl: tupl[3],flatten_list_paths))

    all_labels_types=np.unique(annot['labelName'].to_numpy())
    for targetLab in all_labels_types :
        out_files_frame[targetLab]=list(map(lambda tupl: getLabelPathOrEmpty(targetLab,tupl),flatten_list_paths))
    out_files_frame.to_csv(resCSVDir) 
    
    return out_files_frame




# out_files_frame.columns

# len(allPaths[0])


# flatten_list[0][4]

# allPaths[0][0][0]

# allPaths


# ## get single study
# for current_study_id in np.unique(annot['StudyInstanceUID'].to_numpy()):
    



#/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.8.15.100960.30000022021714130657000003016/U_OjmNZ8/1.3.12.2.1107.5.8.15.100960.30000022021714130657000003047/volume.mha
#/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.8.15.100960.30000022021714463775000001099/U_OjmNZ8/1.3.12.2.1107.5.8.15.100960.30000022021714463775000001281/volume.mha

#/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.8.15.100960.30000022021714130657000000014/U_8ZmM76/1.3.12.2.1107.5.8.15.100960.30000022021714130657000001555/volume.mha




# # Simplify table
# columns_brief = ['id', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'data', 'annotationMode']
# annots_df = annot[columns_brief]

# # Box annotations
# boxes = annots_df[annots_df.annotationMode == 'bbox']

# # Extract box data
# def extract_box_data(df):
#     j = df.copy()
#     j = j[(j.annotationMode == 'bbox') & (~j.data.isnull())]
#     try:
#         j['data'] = j['data'].apply(lambda x:json.loads(x.replace("'", "\"")))
#     except:
#         j['data']

#     j['x'] = [d['x'] for _,d in j.data.iteritems()]
#     j['y'] = [d['y'] for _,d in j.data.iteritems()]
#     j['w'] = [d['width'] for _,d in j.data.iteritems()]
#     j['h'] = [d['height'] for _,d in j.data.iteritems()]
#     j = j.drop('data', axis=1)
#     return j

# boxes = extract_box_data(boxes)
# boxes.head()

# np.unique(boxes['SOPInstanceUID'].to_numpy())
# np.unique(boxes['SeriesInstanceUID'].to_numpy())