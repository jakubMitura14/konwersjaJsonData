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
import shutil
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
from mainFuncs import getLabelsAbbrev
from dicomSeg.dicomSeg import save_dicom_seg_label



def createAnatLabel(annot_for_series,lab,data,paths_dict,series_file_names,image3D,masterolds,currentStudyDesc,jsonFolder,dicomSPath,seriesId):
    """
    create a file and necessery folders for anatomical labels
    in case of the 
    
    """
    if(currentStudyDesc=='t2w'):
        name = f"{str(masterolds)}_{getLabelsAbbrev(lab)}_{currentStudyDesc}"    
        labelNiiPath = join(paths_dict['anat_path_no_seg'],  f"{name}.nii.gz")
        # we will create a folder for the dicom seg buut we will not yet write them as labels requires some processing
        labelSegPath = join(paths_dict['anat_path_seg'],name+'.dcm')
        # os.makedirs(labelSegPath ,exist_ok = True)
        return createLabelFile(annot_for_series,lab,data,labelNiiPath,series_file_names,image3D,labelSegPath,jsonFolder,dicomSPath,seriesId,lab)
    return (' ',' ',' ',' ',' ',' ',' ')
def createLesionLabel(annot_for_series,lab,data,paths_dict,series_file_names,image3D,masterolds,docId,currentStudyDesc,jsonFolder,dicomSPath,seriesId):
    """
    create a file and necessery folders for anatomical labels
    in case of the     
    """
    labb = lab.replace(" ","")
    name = f"{str(masterolds)}_{labb}_{docId}_{currentStudyDesc}"
    os.makedirs(join(paths_dict['lesion_path_no_seg'],labb) ,exist_ok = True)
    # os.makedirs(join(paths_dict['lesion_path_seg'],labb) ,exist_ok = True)

    lesionNiiPath = join(paths_dict['lesion_path_no_seg'],labb,name+'.nii.gz')
    # we will create a folder for the dicom seg buut we will not yet write them as labels requires some processing
    lesionlSegPath = join(paths_dict['lesion_path_seg'],labb,name+'.dcm').strip()
    # os.makedirs(lesionlSegPath ,exist_ok = True)
    return createLabelFile(annot_for_series,f"{labb}_{docId}_{currentStudyDesc}",data,lesionNiiPath,series_file_names,image3D,lesionlSegPath,jsonFolder,dicomSPath,seriesId ,labb)


def createLabelFile(annot_for_series,lab,data,labelNiiPath,series_file_names,image3D,labelSegPath,jsonFolder,dicomSPath,seriesId,labb):
    """
    takes label dataframe and original MRI image and recreates label volume from it 
    """
    dtype=np.uint16
    annot_for_label=annot_for_series.loc[annot_for_series['labelName'] == lab]
    zeroArray=np.zeros(data.shape, dtype=dtype)

    writer = sitk.ImageFileWriter()
    if(not pathOs.exists(labelNiiPath)):
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
        try:
            image.SetDirection(image3D.GetDirection())    
        except:
            print(f"error saving direction of {labelNiiPath}")
            
        writer.SetFileName(labelNiiPath)
        writer.Execute(image)
        save_dicom_seg_label([],jsonFolder,' ',  labelSegPath,dicomSPath,seriesId,image,labb)

        
    return (getLabelsAbbrev(lab),labelNiiPath,labelSegPath)

def translateSeriesDesc(series_desc_string,acqNumb):
    """
    changes unintuitive series description tag into more human readable format 
    using manually set list of tuples where first entry is the original series tag
    and second entry human readable version
    what is important sometimes multipla tags will map to the same human readable one
    """
    acqNumb=acqNumb.replace("'","")
    manual_map_list=[
        ("'t2_bl_tse_tra_sFOV'"                             ,"t2w")
        ,("'t2_bl_tse_tra_sFOV'"                            ,"t2w")
        ,("'t2_bl_tse_tra_p'"                               ,"t2w")
        ,("'[KOELIS] t2_bl_tse_tra_p'"                      ,"t2w")
        ,("'t2_bl_tse_fs_tra_WIDE'"                         ,"t2w")
        ,("'ep2d_diff_b 50 400 800 1200_TRACEW'"            ,"hbv")
        ,("'ep2d_diff_b 50 400 800 1200_ADC'"               ,"adc")    
        ,("'t1_fl3d_tra fs_dyn CM'"                         ,f"dce{acqNumb}") 
        ,("'t1_fl3d_tra fs_dyn CM_PEI'"                     ,f"dce{acqNumb}") 
        ,("'t1_fl3d_tra fs_dyn CM_SUB_MOCO'"                ,f"dce{acqNumb}") 
        ,("'t1_fl3d_tra fs_dyn CM_SUB_MIP_COR'"             ,f"dce{acqNumb}") 
        ,("'t1_fl3d_tra fs_dyn CM_SUB'"                     ,f"dce{acqNumb}") 
        ,("'t1_fl3d_tra_dyn CM'"                            ,f"dce{acqNumb}") 
        ,("'t1_fl3d_tra_dyn CM_PEI'"                        ,f"dce{acqNumb}") 
        ,("'t1_fl3d_tra_dyn CM_PEI'"                        ,f"dce{acqNumb}")       
        ,("'t2_bl_tse_tra_P'"                               ,"t2w")
        ,("'t2_bl_tse_fs_tra'"                              ,"t2w")
        ,("'t2_bl_tse_tra'"                                 ,"t2w")
        ,("'t2_bl_tse_cor'"                                 ,"cor")
        ,("'t2_bl_tse_COR_P'"                               ,"cor")
        ,("'t2_bl_tse_sag'"                                 ,"sag")
        ,("'t2_bl_tse_SAG_P'"                               ,"sag")
        
        ]
    #    manual_map_list=[
    #     ("'t2_bl_tse_tra_sFOV'"                   , "t2_transverse"  )
    #     ,("'t2_bl_tse_tra_p'"                   , "t2_transverse"  )
    #     ,("'ep2d_diff_b 50 400 800 1200_TRACEW'"            , "dwi_transverse"  )
    #     ,("'ep2d_diff_b 50 400 800 1200_ADC'"               ,"adc_transverse"   )    
    #     ,("'t1_fl3d_tra fs_dyn CM'"                         , "t1_dce_transverse"  ) 
    #     ,("'t2_bl_tse_tra_P'"                               , "t2_transverse"  )
    #     ,("'t2_bl_tse_tra'"                                 ,  "t2_transverse" )
    #     ,("'t2_bl_tse_cor'"                                 , "t2_coronal"  )
    #     ,("'t2_bl_tse_sag'"                                 , "t2_saggital"  )]
    res=list(filter(lambda tupl: tupl[0]==series_desc_string ,manual_map_list))
    if(len(res)==0):
        print(f" series description string {series_desc_string} acq numb {acqNumb} not known please adjust translateSeriesDesc function in getDirAndnumbFrame file")
        return f"unknown{series_desc_string}".strip()
    return res[0][1]

def getFirstOrEmpty(arr):
    if(len(arr)>0):
        return arr[0]
    return ' '    



def getSeriesPathsAndMasterNumb(files_df,files_df_origFolds, currentSeries,masterolds_in_Study):
    """
    get list of paths in original data - There may be still duplicates in output !!!
    apart from it we will get the master number associated to this series
    Also it will give back the proper name of the modality
    """
    ### 1) now we will get single series - get all paths of files related to it
    locDf = files_df.loc[files_df['SeriesInstanceUID'] == currentSeries]
    locDfB = files_df_origFolds.loc[files_df_origFolds['SeriesInstanceUID'] == currentSeries]
    locDf=locDf.drop_duplicates(subset = ["SOPInstanceUID"])
    # in order to maximise the chance that we will getaround tag problems we will use dicoms from both sources
    paths_in_series= np.unique(locDf['paths'].to_numpy())
    paths_in_seriesB= np.unique(locDfB['paths'].to_numpy())
    paths_in_series=[*paths_in_series,*paths_in_seriesB]
    # masterolds_in_series= np.unique(locDf['masterolds'].to_numpy())
    masterolds=masterolds_in_Study#(masterolds_in_series[0]).replace('nas-lssi-dco/','')
    #additionally we will get the study description and acqisition number which are needed for study type description
    dses=list(map(pydicom.dcmread ,paths_in_series))
    studeDescs=(list(map(mainFuncs.get_SeriesDesc  ,dses)))
    studeDescs=(list(filter(lambda el: el!=' ',studeDescs)))
    studeDescs=np.unique(studeDescs)
    #acquisition number
    acqNumbs=(list(map(mainFuncs.get_Aquisition_Number  ,dses)))
    acqNumbs=(list(filter(lambda el: el!=' ',acqNumbs)))
    acqNumbs=np.unique(acqNumbs)  
    currentStudyDesc=translateSeriesDesc(getFirstOrEmpty(studeDescs),getFirstOrEmpty(acqNumbs))

    return (paths_in_series,masterolds,currentStudyDesc)

def getSliceAndSOP(filename):
    """
    finds SOP uid of the given dicom file
    """
    ds = pydicom.dcmread(filename)
    res=(' ',' ', ' ')
    try:
        res= (mainFuncs.get_SOPInstanceUID(ds),ds.pixel_array ,filename)
    except:
        print("error reading slice")    
    return res

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


def findTheSameNoAnnot(slice,slices_with_sop):
    """
    used in order to be sure that we do not have two slices with the same data
    """
    # we look for all slices holding the same data
    theSame= list(filter(lambda tupl : np.array_equal(tupl[1],slice)  ,slices_with_sop))
    return theSame[0]    

def getUniqPaths(paths_in_series,annot_for_series ):
    """
    given the list of paths to the diffrent slice it return back only those that are unique 
    Important is that in the dataset at hand dicoms with the same SOP ids may be repeated as well 
    dicoms with identical data but diffrent sops - in order to deal with this problem one need to manually
    check the pixel array of each slice for equality
    Additional difficulty is related to the fact that in the annotations only some of the SOP ids are referenced
    hence here if the series is annotated we need to be sure that we are returning paths that are present in annotation
    dataframe
    """
    #first we get data from each file and check the sops present in annotation data frame
    slices_with_sop = list(map(getSliceAndSOP ,paths_in_series))
    slices_with_sop = list(filter(lambda el: el[0]!=' ' ,slices_with_sop))
    sops_in_anot= np.unique(annot_for_series['SOPInstanceUID'].to_numpy())
    # if there are any sops in the related annotation data frame - if there are annotations related to this series
    if(len(sops_in_anot)>0):
        filtered = list(map(lambda tupl :  findTheSame(tupl[1],slices_with_sop,sops_in_anot) ,slices_with_sop))
        return np.unique(list(map(lambda tupl: tupl[2],filtered)))
    filtered = list(map(lambda tupl :  findTheSameNoAnnot(tupl[1],slices_with_sop) ,slices_with_sop))
    return np.unique(list(map(lambda tupl: tupl[2],filtered)))

def saveMainMRI(paths_in_series,paths_dict,currentStudyDesc,currentSeries):
    """
    given paths to  dicom files with unique slices we will copy those into separate dicom folder and 
    generate a single mha file 
    """
    origVolPath=join(paths_dict['data_path_seg'],currentStudyDesc)
    os.makedirs(origVolPath ,exist_ok = True)
    pathMha=join(paths_dict['data_path_no_seg'],f"{currentStudyDesc}.mha")
    
    image3D=[]
    series_file_names=[]
    if(not pathOs.exists(pathMha)):
        try:
            for path_to_copy in paths_in_series:
                shutil.copyfile(path_to_copy,join(origVolPath,Path(path_to_copy).name) )
            # print(f"origVolPath {origVolPath}")
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(origVolPath)
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(origVolPath, currentSeries)
            series_reader = sitk.ImageSeriesReader()
            series_reader.MetaDataDictionaryArrayUpdateOn()
            series_reader.LoadPrivateTagsOn()
            series_reader.SetFileNames(series_file_names)

            image3D = series_reader.Execute()
            writer = sitk.ImageFileWriter()
            # Use the study/series/frame of reference information given in the meta-data
            # dictionary and not the automatically generated information from the file IO
            writer.SetFileName(pathMha)
            writer.Execute(image3D)   
        except Exception as err:
            print(f"error in saveMainMRI {currentStudyDesc} {currentSeries} pathh {pathMha} {err}")   
            return origVolPath,pathMha,' ',' ',' '
    else:
        #no point in recreating if it already exist
        image3D=sitk.ReadImage(pathMha)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(origVolPath,currentSeries)

    return origVolPath,pathMha,sitk.GetArrayFromImage(image3D),image3D,series_file_names



def mainGenereteFiles(files_df,files_df_origFolds,annot_for_series,files_for_series,currentSeries,current_study_id,mainPaths_studyId,masterolds_in_Study,jsonFolder):
    """
    main functions that gives single series of single annotator specialist
    will create files with 3d mha of this series plus 3d nii.gz files for each annotation label
    return current_study_id current_doctor_id series number, path to the main volume and list of tuples containing label name and 
    """
    #get all dicom paths associated with series including duplicates    
    paths_in_series,masterolds,currentStudyDesc=getSeriesPathsAndMasterNumb(files_df,files_df_origFolds, currentSeries,masterolds_in_Study)
    masterolds=masterolds_in_Study
    if(currentStudyDesc!="PhoenixZIPReport"):
        
        #prepare unique paths 
        paths_in_series=getUniqPaths(paths_in_series,annot_for_series )

        #dict with paths of main folders needed here
        paths_dict = dict((x, y) for x, y in mainPaths_studyId)
        ## saving current series MRI both as mha file and as folder with dicoms - where we store only unique dicoms
        origVolPath,pathMha,image3Ddata,image3D,series_file_names = saveMainMRI(paths_in_series,paths_dict,currentStudyDesc,currentSeries)
        # in case of error in the main image reading we will not return anything
        if(image3Ddata==' '):
            return (' ',' ',' ',' ',' ',' ' ,' ' )    
        
        ## saving anatomical labels (if they exist)
        #we will check how many diffrent labels are associated in case of no annotations it will be empty
        uniq_labels= np.unique(annot_for_series['labelName'].to_numpy())
        uniqLabelsAnatomy = list(filter(lambda lab: 'lesion' not in lab,uniq_labels))
        uniqLabelsLesions = list(filter(lambda lab: 'lesion' in lab,uniq_labels))

        labelNameAndPaths=[]

        if(len(uniqLabelsAnatomy)>0):
            labelNameAndPaths=list(map(lambda lab: createAnatLabel(annot_for_series,lab,image3Ddata,paths_dict,series_file_names,image3D,masterolds,currentStudyDesc,jsonFolder,origVolPath,currentSeries),uniqLabelsAnatomy ))
        labelNameAndPaths=list(filter(lambda el: el[0]!=' ',labelNameAndPaths))
        # saving data about lesions separately for each doctor (annotator)
        if(len(uniqLabelsLesions)>0):
            for lab in uniqLabelsLesions:
                for docId in np.unique(annot_for_series['createdById'].to_numpy()):
                    annot_for_doctor=annot_for_series.loc[annot_for_series['createdById'] == docId]
                    res = createLesionLabel(annot_for_doctor,lab,image3Ddata,paths_dict,series_file_names,image3D,masterolds,docId,currentStudyDesc,jsonFolder,origVolPath,currentSeries )
                    labelNameAndPaths.append(res)


        # newPath= os.path.join(copiedPath,'volume.mha')
        # series_file_names=None
        # image3D=None
        # #avoiding creating file if one is present
        # if(not pathOs.exists(newPath)): 

        #     os.makedirs(origVolPath ,exist_ok = True)
        #     # into each subfolder we will copy the full  set of files related to main image at hand
        #     for path_to_copy in paths_in_series:
        #         os.system(f'cp {path_to_copy} {origVolPath}') 


        #     series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(origVolPath)
        #     series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(origVolPath, series_IDs[0])

        #     #getseries file names in correct order
        #     #series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(copiedPath, currentSeries)
        #     series_reader = sitk.ImageSeriesReader()
        #     series_reader.MetaDataDictionaryArrayUpdateOn()
        #     series_reader.LoadPrivateTagsOn()
        #     series_reader.SetFileNames(series_file_names)

        #     image3D = series_reader.Execute()
        #     writer = sitk.ImageFileWriter()
        #     # Use the study/series/frame of reference information given in the meta-data
        #     # dictionary and not the automatically generated information from the file IO
        #     writer.SetFileName(newPath)
        #     writer.Execute(image3D)   
        #     print(f"newPath image3D {newPath}")
        # else:
        #     image3D=sitk.ReadImage(newPath)
        #     series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(origVolPath)
        #     series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(origVolPath, series_IDs[0])

        # data=sitk.GetArrayFromImage(image3D)


        return (current_study_id,currentSeries,currentStudyDesc,pathMha,origVolPath,labelNameAndPaths ,masterolds )
    return (' ',' ',' ',' ',' ',' ' ,' ' )
        
def createStudyFolder(item,masterolds ): 
    key,value=item
    masteroldsStand=str(masterolds)
    if(len(masteroldsStand)==1):
        masteroldsStand=f"00{masteroldsStand}"
    if(len(masteroldsStand)==2):
        masteroldsStand=f"0{masteroldsStand}"
    newPath = join(value,masteroldsStand)    
    os.makedirs(newPath ,exist_ok = True)
    return (key,newPath)

def iterate_overStudy(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder):
    """
    iterate ove all series with the same study UID
    """
    res=[]
    annot_for_study_id=annot.loc[annot['StudyInstanceUID'] == current_study_id]
    files_for_study_id=files_df_origFolds.loc[files_df_origFolds['StudyInstanceUID'] == current_study_id]
    # 
    #get annotator id 
    # for current_doctor_id in np.unique(annot_for_study_id['createdById'].to_numpy()):
    #     annot_for_doctor=annot_for_study_id.loc[annot_for_study_id['createdById'] == current_doctor_id]
        # #create directory for this study
        # studyPath = os.path.join(outputDir, current_study_id,current_doctor_id)
        # os.makedirs(studyPath, exist_ok = True)
        #get single series
    aa=sorted(files_df_origFolds['masterolds'].to_numpy(),key = lambda el: len(el))[0]

    masterolds_in_Study=files_for_study_id['masterolds'].to_numpy()
    # print(f"ffffffffff {aa}")    
    # print(f"ffffffffff **{masterolds_in_Study[0]}**")    
    masterolds=(masterolds_in_Study[0]).replace('nas-lssi-dco/','')
    if(masterolds==' '):
        masterolds=f"unknownMasterNum_{current_study_id}"
        print("unknownnn masterrr ")
    mainPaths_studyId=list(map(partial(createStudyFolder,masterolds=masterolds),mainPaths.items()))
    # print(f"mainPaths_studyId {mainPaths_studyId}")

    for currentSeries in np.unique(files_for_study_id['SeriesInstanceUID'].to_numpy()):
        annot_for_series=annot_for_study_id.loc[annot_for_study_id['SeriesInstanceUID'] == currentSeries]
        files_for_series=files_for_study_id.loc[files_for_study_id['SeriesInstanceUID'] == currentSeries]

        res.append(mainGenereteFiles(files_df,files_df_origFolds,annot_for_series,files_for_series
            ,currentSeries,current_study_id,mainPaths_studyId,masterolds,jsonFolder))
    res= np.array(res)
    print(f"iterate_overStudy res {res.shape}  {res[0]}")
    res=list(filter( lambda el: el[0]!=' ' ,res))        
    return res


def iterate_overStudySafe(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder):
    """
    just gives one restart of iterate_overStudy function in case of failure
    """
    try:
        return iterate_overStudy(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder)
    except:
        return iterate_overStudy(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder)


def getLabelPathOrEmpty(targetLab, tupl,pathIndex):
    """
    return path to the label file if it exists otherwise " "
    """
    listLabs=tupl[5]
    for labb in listLabs :
        if(labb[0]== targetLab):
            return labb[pathIndex]
    return " "    

def getLabelNames(tupl):
    """
    return path to the label file if it exists otherwise " "
    """
    listLabs=tupl[5]
    # for labb in listLabs :
    #     labb[0]
    res= list(map(lambda entry: entry[0],listLabs ))   
    print(f"getLabelNames {res}")
    return res



def get_frame_with_output(files_df,files_df_origFolds,annot,outputDir,resCSVDir,mainFoldDirMha,mainFoldDirSeg,jsonFolder):
    """
    in parallel iterates over all studies and series and save the paths of created files in the csv file
    """
    if(pathOs.exists(resCSVDir)):
        return pd.read_csv(resCSVDir) 


    out_files_frame= pd.DataFrame()


    # dicomSeg main folders
    data_path_seg= join(mainFoldDirSeg,"Data")
    anat_path_seg= join(mainFoldDirSeg,"Anatomical_Labels")
    lesion_path_seg= join(mainFoldDirSeg,"Lesion_Labels")

    #mha and Nifti main folders
    data_path_no_seg= join(mainFoldDirMha,"Data")
    anat_path_no_seg= join(mainFoldDirMha,"Anatomical_Labels")
    lesion_path_no_seg= join(mainFoldDirMha,"Lesion_Labels")
    #creating all required folders given they do not exist already
    list(map(lambda pathh: os.makedirs(pathh ,exist_ok = True)  ,[data_path_seg,anat_path_seg,lesion_path_seg,data_path_no_seg,anat_path_no_seg,lesion_path_no_seg ]))
    #dict giving access to main directories
    mainPaths = {'data_path_seg':data_path_seg,'anat_path_seg':anat_path_seg,'lesion_path_seg':lesion_path_seg
                 , 'data_path_no_seg':data_path_no_seg , 'anat_path_no_seg':anat_path_no_seg , 'lesion_path_no_seg':lesion_path_no_seg }



    #iterate over all files
    allPaths=[]
    with mp.Pool(processes = mp.cpu_count()) as pool: 
         allPaths=pool.map(partial(iterate_overStudySafe, files_df=files_df,files_df_origFolds=files_df_origFolds,annot=annot,outputDir=outputDir,mainPaths=mainPaths,jsonFolder=jsonFolder), np.unique(files_df_origFolds['StudyInstanceUID'].to_numpy()))
    #allPaths=list(map(partial(iterate_overStudySafe, files_df=files_df,files_df_origFolds=files_df_origFolds,annot=annot,outputDir=outputDir,mainPaths=mainPaths), np.unique(files_df_origFolds['StudyInstanceUID'].to_numpy())))


    flatten_list_paths = list(itertools.chain(*allPaths))
    out_files_frame['study_id']=list(map(lambda tupl: tupl[0],flatten_list_paths))
    # out_files_frame['doctor_id']=list(map(lambda tupl: tupl[1],flatten_list_paths))
    out_files_frame['series_id']=list(map(lambda tupl: tupl[1],flatten_list_paths))
    out_files_frame['series_desc']=list(map(lambda tupl: tupl[2],flatten_list_paths))
    out_files_frame['series_MRI_path']=list(map(lambda tupl: tupl[3],flatten_list_paths))
    out_files_frame['dicom_MRI_path']=list(map(lambda tupl: tupl[4],flatten_list_paths))
    out_files_frame['masterolds']=list(map(lambda tupl: tupl[6],flatten_list_paths))

    all_labels_types=list(map(lambda el: getLabelNames(el),flatten_list_paths))
    all_labels_types=list(itertools.chain(*all_labels_types))
    print(f"all_labels_types {all_labels_types}")
    all_labels_types=np.unique(all_labels_types)
    print(f"bbb all_labels_types {all_labels_types}")

    for targetLab in all_labels_types :
        out_files_frame[f"{targetLab}_noSeg"]=list(map(lambda tupl: getLabelPathOrEmpty(targetLab,tupl,1),flatten_list_paths))
        out_files_frame[f"{targetLab}_Seg"]=list(map(lambda tupl: getLabelPathOrEmpty(targetLab,tupl,2),flatten_list_paths))



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