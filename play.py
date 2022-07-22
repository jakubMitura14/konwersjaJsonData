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

exampleRow=list(results['annotations'].iterrows())[0]
binArray= mainFuncs.load_mask_instance(exampleRow[1])
binArray.shape

#path to folder with all required data
dataDir='/workspaces/konwersjaJsonData/data'
outputDir='/workspaces/konwersjaJsonData/output'
files_df= mainFuncs.get_df_file_info(dataDir)

# #adding columns to dataframe collecting information about outputs paths
# uniq_labels= np.unique(annot['labelName'].to_numpy())
# uniq_labels_and_vol=np.concatenate((uniq_labels,['volume']))

# for lab in uniq_labels_and_vol:
#     annot[lab]=''

## get single study
current_study_id=np.unique(annot['StudyInstanceUID'].to_numpy())[0]


annot_for_study_id=annot.loc[annot['StudyInstanceUID'] == current_study_id]

#get annotator id 
current_doctor_id=np.unique(annot_for_study_id['createdById'].to_numpy())[0]
annot_for_doctor=annot_for_study_id.loc[annot_for_study_id['createdById'] == current_doctor_id]

#create directory for this study
studyPath = os.path.join(outputDir, current_study_id,current_doctor_id)
os.makedirs(studyPath, exist_ok = True)


#get single series
example_series_id=np.unique(annot_for_doctor['SeriesInstanceUID'].to_numpy())[0]


currentSeries=example_series_id
annot_for_series=annot_for_doctor.loc[annot_for_doctor['SeriesInstanceUID'] == currentSeries]

### 1) now we will get single series - get all paths of files related to it
paths_in_series= files_df.loc[files_df['SeriesInstanceUID'] == currentSeries]['paths'].to_numpy() 
### 2) we will check how many diffrent labels are associated 
uniq_labels= np.unique(annot_for_series['labelName'].to_numpy())
uniq_labels_and_vol=np.concatenate((uniq_labels,['volume']))

#adding folder for each label and pass all files into it
# for each label we will generate searate subfolder
for lab in uniq_labels_and_vol:
    locPath=os.path.join(studyPath,currentSeries ,lab.replace(" ","_"))
    out_paths=[]
    os.makedirs(locPath ,exist_ok = True)
    # into each subfolder we will copy the full  set of files related to main image at hand
    for path_to_copy in paths_in_series:
        os.system(f'cp {path_to_copy} {locPath}')  
    


### we will change the data type of each file into boolean and fil it with zeros
#first get all files paths - but not of the original file
completePaths= []
for lab in uniq_labels_and_vol:
    locPath=os.path.join(studyPath,currentSeries ,lab.replace(" ","_"))
    locPaths=mainFuncs.get_all_file_paths(locPath)
    dtype=np.uint16
    completePaths=completePaths+locPaths
    if(lab!='volume'):
        annot_for_label=annot_for_series.loc[annot_for_series['labelName'] == lab]
        for loccPath in locPaths:
            #open file to get sop id
            ds = pydicom.dcmread(loccPath)
            sop=mainFuncs.get_SOPInstanceUID(ds)
            annot_for_sop= annot_for_label.loc[annot_for_label['SOPInstanceUID'] == sop]       
            if(len(annot_for_sop)>0):
                print("overWriting")
                rowOfIntr=list(annot_for_sop.iterrows())[0]
                #we obtain mask as a boolean array
                binArray= mainFuncs.load_mask_instance(rowOfIntr).astype(dtype)  
                #time to overwrite the data
                ds.PixelData = binArray.tostring() 
 
                ds.save_as(loccPath)
                
                # img=sitk.ReadImage(loccPath)
                # #data=sitk.GetArrayFromImage(image1))
                # #recreating image keeping relevant metadata
                # img2 = sitk.GetImageFromArray(binArray)  
                # for key in img.GetMetaDataKeys(): 
                #     img2.SetMetaData(key, img.GetMetaData(key))           
                # writer = sitk.ImageFileWriter()
                # # writer.KeepOriginalImageUIDOn()
                # writer.SetFileName(loccPath)
                # writer.Execute(img2)
            else: # for case we have no data we will just overwrite the pixel data
                img=sitk.ReadImage(loccPath)
                data=sitk.GetArrayFromImage(img)

                binArray=np.zeros(data.shape, dtype=dtype)
                ds.PixelData = binArray.tostring() 
                ds.save_as(loccPath)

                # #recreating image keeping relevant metadata
                # img2 = sitk.GetImageFromArray(binArray)  
                # for key in img.GetMetaDataKeys(): 
                #     img2.SetMetaData(key, img.GetMetaData(key))           
                # writer = sitk.ImageFileWriter()
                # # writer.KeepOriginalImageUIDOn()
                # writer.SetFileName(loccPath)
                # writer.Execute(img2)          
    # data is already overwritten
    # reading series of dicom and save them as nii.gz in case of the 
    # from https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html
    #series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(locPath)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
    locPath, currentSeries)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(locPaths)
    # series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()
    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()
    newPath= os.path.join(locPath,lab+'.nii.gz')
    if(lab=='volume'):
        newPath= os.path.join(locPath,'volume.mha')
    print("nnnnnnn "+newPath)
    writer.SetFileName(newPath)
    writer.Execute(image3D)    


# for lab in uniq_labels_and_vol:
#     locPath=os.path.join(studyPath,currentSeries ,lab.replace(" ","_"))

pathhg='/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.2.41.69644.30000020011405272273800000001/U_8ZmM76/1.3.12.2.1107.5.2.41.69644.2020011406404167573202555.0.0.0/volume/volume.mha'
img=sitk.ReadImage(pathhg)
data=sitk.GetArrayFromImage(img)
# exampleRow=list(mergedWithPaths.iterrows())[0]



#now we associate the path with proper row in annotation dataframe


### for slices where annotations are present we will ovewrite data in file with data from dataframe
#we need to change data type of each voxel to boolean and set it to zeros

ds = pydicom.dcmread(somePath)
pxx= ds.pixel_array
pxx.shape
somePath=completePaths[1]



#for labPath in completePaths:



###  we will load all files from each subfolder using simpleitk and save them in case of main image as mha in case of labels as nii.gz




files_df #438554 rows
annot #15975

mergedWithPaths=pd.merge(annot,files_df,on='SOPInstanceUID',how='inner')
mergedWithPaths.iloc()[1]

numpy_StudyInstanceUID=annot['StudyInstanceUID'].to_numpy()
len(np.unique(numpy_StudyInstanceUID))

mergedWithPaths=pd.merge(annot,files_df,on='SOPInstanceUID',how='inner')
mergedWithPaths.columns

parsedd = list(map(mainFuncs.load_mask_instance  ,list(mergedWithPaths.iterrows())))
filtered= list(filter(lambda it : np.sum(it) >0,parsedd))
len(filtered)


example_study_id=np.unique(annot['StudyInstanceUID'].to_numpy())[0]

annots_for_study_id= annot.loc[annot['StudyInstanceUID'] == example_study_id]
files_for_study_id=files_df.loc[files_df['StudyInstanceUID'] == example_study_id]

#get single series
example_series_id=np.unique(annots_for_study_id['SeriesInstanceUID'].to_numpy())[0]

annots_for_series_id= annots_for_study_id.loc[annots_for_study_id['SeriesInstanceUID'] == example_series_id]
files_for_series_id=files_for_study_id.loc[files_for_study_id['SeriesInstanceUID'] == example_series_id]


#get sop
lenn= len(np.unique(files_for_series_id['SOPInstanceUID'].to_numpy()))
lenn

for i in range(0,lenn-1):
    example_sop_id=np.unique(annots_for_series_id['SOPInstanceUID'].to_numpy())[i]

    annots_for_sop_id= annots_for_series_id.loc[annots_for_series_id['SOPInstanceUID'] == example_sop_id]
    files_for_sop_id=files_for_series_id.loc[files_for_series_id['SOPInstanceUID'] == example_sop_id]
    print(f" annots_for_sop_id {len(annots_for_sop_id)} files_for_sop_id {len(files_for_sop_id)}  ")


annots_for_sop_id
files_for_sop_id

files_for_sop_id['paths'].to_numpy()[0]
files_for_sop_id['paths'].to_numpy()[1]

'1.3.12.2.1107.5.2.41.69644.2020011406404167573202555.0.0.0/1.3.12.2.1107.5.2.41.69644.2020011406404215288402586'=='1.3.12.2.1107.5.2.41.69644.2020011406404167573202555.0.0.0/1.3.12.2.1107.5.2.41.69644.2020011406404215288402586'

files_for_sop_id.columns

annots_for_study_id
files_for_study_id

# Get DICOM pixel array
# pixel_array = mdai.visualize.load_dicom_image(image_id, to_RGB=False, rescale=True)

# number_of_masks = len(mask[1])
# fig = plt.figure()
# for i in range(0,number_of_masks):
#     cols = 3
#     rows = np.ceil(number_of_masks/float(cols))
#     ax = fig.add_subplot(rows, cols, i + 1)
#     ax.axis('off')
#     plt.imshow((cv2.bitwise_and(img, img, mask = mask[0][:,:,i].astype(np.uint8))))
#     ax.set_title(mask[1][i])


someFile='/workspaces/konwersjaJsonData/data/054/DICOM/22021715/11310000/39978953'
someFileB='/workspaces/konwersjaJsonData/data/054/DICOM/22021715/11310000/39984365'
ds = pydicom.dcmread(someFileB)


singleSeries='/workspaces/konwersjaJsonData/locData/055/DICOM/22022414/23560000/39048749'
single_series_files_df= mainFuncs.get_df_file_info(singleSeries)

exampleRow=list(mergedWithPaths.iterrows())[0]
binArray= mainFuncs.load_mask_instance(exampleRow)



# resList=[]
# with mp.Pool(processes = mp.cpu_count()) as pool:
#     resList=pool.map(mainFuncs.load_mask_instance  ,list(mergedWithPaths.iterrows()))



#SOPInstanceUID 31950
#SeriesInstanceUID 1485150
#StudyInstanceUID 24252720



## now performing outer join
#from annotations
#StudyInstanceUID - 199 unique
#SeriesInstanceUID - 743 unique
#from files
#StudyInstanceUID - 3 unique
#SeriesInstanceUID - 44 unique