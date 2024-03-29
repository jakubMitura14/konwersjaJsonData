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
    # if(currentStudyDesc!='t2w'):
    #     print(f"ccccccreateAnatLabel {lab} currentStudyDesc {currentStudyDesc} ")
    # abb=getLabelsAbbrev(lab)
    # is_new_anatom=( (abb=='bladder_lumen') or (abb=='bladder_wall') or (abb=='rec_abd_L') or abb=='rec_abd_R')

    if(currentStudyDesc=='t2w'):
        name = f"{str(masterolds)}_{getLabelsAbbrev(lab)}_{currentStudyDesc}"    
        labelNiiPath = join(paths_dict['anat_path_no_seg'],  f"{name}.nii.gz")
        # we will create a folder for the dicom seg buut we will not yet write them as labels requires some processing
        labelSegPath = join(paths_dict['anat_path_seg'],name+'.dcm')
        # os.makedirs(labelSegPath ,exist_ok = True)
        return createLabelFile(annot_for_series,lab,data,labelNiiPath,series_file_names,image3D,labelSegPath,jsonFolder,dicomSPath,seriesId,lab,lab)
    return (' ',' ',' ',' ',' ',' ',' ')


def createLesionLabel(annot_for_series,lab,data,paths_dict,series_file_names,image3D,masterolds,docId,currentStudyDesc,jsonFolder,dicomSPath,seriesId,corrections_for_doc):
    """
    create a file and necessery folders for anatomical labels
    in case of the     
    """
    labb = lab.replace(" ","")
    # now we need to use corrections_for_doc to check weater the lesion should be saved at all and whether its number is correct beware that lesion names in corrections_for_doc is just number for example 1 not lesion1
    locDfCorr = corrections_for_doc.loc[corrections_for_doc['lesion_label'] == int(labb.replace('lesion','')) ]
    if(len(locDfCorr)==0):
        print(f"NNnno Label in correction df masterolds {masterolds} docId {docId}  labb {labb}")
        return (' ', ' ', ' ', 0)
    newLabNum=locDfCorr['lesion_id'].to_numpy()[0]
    # to_remove=locDfCorr['Do_usun'].to_numpy()[0]

    # to_remove= ((to_remove=='True' or to_remove))
    labb= f"lesion{newLabNum}"
    # to_remove= ((to_remove=='True' or to_remove))
    # if(not to_remove):
    origLab=lab
    name = f"{str(masterolds)}_{labb}_{currentStudyDesc}_{docId}"
    os.makedirs(join(paths_dict['lesion_path_no_seg'],labb) ,exist_ok = True)
    os.makedirs(join(paths_dict['lesion_path_no_seg'],labb,currentStudyDesc) ,exist_ok = True)
    os.makedirs(join(paths_dict['lesion_path_seg'],labb) ,exist_ok = True)
    os.makedirs(join(paths_dict['lesion_path_seg'],labb,currentStudyDesc) ,exist_ok = True)

    lesionNiiPath = join(paths_dict['lesion_path_no_seg'],labb,currentStudyDesc,name+'.nii.gz')
    # we will create a folder for the dicom seg buut we will not yet write them as labels requires some processing
    lesionlSegPath = join(paths_dict['lesion_path_seg'],labb,currentStudyDesc,name+'.dcm').strip()
    # os.makedirs(lesionlSegPath ,exist_ok = True)
    return createLabelFile(annot_for_series,f"{labb}_{docId}_{currentStudyDesc}",data,lesionNiiPath,series_file_names,image3D,lesionlSegPath,jsonFolder,dicomSPath,seriesId ,labb,origLab)
    
    

def createNew_anatomy_label(annot_for_series,lab,data,paths_dict,series_file_names,image3D,masterolds,docId,currentStudyDesc,jsonFolder,dicomSPath,seriesId,corrections_for_doc):

    # to_remove= ((to_remove=='True' or to_remove))
    abb=getLabelsAbbrev(lab)
    labb = abb
    # to_remove= ((to_remove=='True' or to_remove))
    # if(not to_remove):
    origLab=lab
    name = f"{str(masterolds)}_{labb}_{currentStudyDesc}_{docId}"
    os.makedirs(join(paths_dict['anat_path_no_seg'],labb) ,exist_ok = True)
    os.makedirs(join(paths_dict['anat_path_no_seg'],labb,currentStudyDesc) ,exist_ok = True)
    os.makedirs(join(paths_dict['anat_path_seg'],labb) ,exist_ok = True)
    os.makedirs(join(paths_dict['anat_path_seg'],labb,currentStudyDesc) ,exist_ok = True)

    lesionNiiPath = join(paths_dict['anat_path_no_seg'],labb,currentStudyDesc,name+'.nii.gz')
    # we will create a folder for the dicom seg buut we will not yet write them as labels requires some processing
    lesionlSegPath = join(paths_dict['anat_path_seg'],labb,currentStudyDesc,name+'.dcm').strip()
    # os.makedirs(lesionlSegPath ,exist_ok = True)
    return createLabelFile(annot_for_series,f"{labb}_{docId}_{currentStudyDesc}",data,lesionNiiPath,series_file_names,image3D,lesionlSegPath,jsonFolder,dicomSPath,seriesId ,labb,origLab)
    

    
    # return (' ', ' ', ' ', 0)
#krowa najpierw sekwencja potem radiolog i w modalnosciach najpierw numer potem adc  i podfoldery w lesionach jako modalnosci


def add_layer_to_be_consistent(binArray):
    """
    we will here check weather in z direction we have consistent shapes -
        given a layer when there is True both above and below it should be True also in case of this layer
    binArray - analyzed boolean array
    """
    shapee=binArray.shape
    up=binArray[0:shapee[0]-2,:,:]
    # mid=binArray[1:shapee[0]-1,:,:]
    down=binArray[2:shapee[0],:,:]
    consistent_part=np.logical_and(up,down)
    consistent_part=np.pad(consistent_part,((1,1),(0,0),(0,0)))
    return np.logical_or(binArray,consistent_part)

    


def createLabelFile(annot_for_series,lab,data,labelNiiPath,series_file_names,image3D,labelSegPath,jsonFolder,dicomSPath,seriesId,labb,origLab):
    """
    takes label dataframe and original MRI image and recreates label volume from it 
    """
    dtype=np.uint16

    annot_for_label=annot_for_series.loc[annot_for_series['labelName'] == origLab]
    # print(f"1 lllllll lab {lab} ")

    zeroArray=np.zeros(data.shape, dtype=dtype)

    writer = sitk.ImageFileWriter()
    # if(not pathOs.exists(labelNiiPath)):
    if(True):
        for index,loccPath in enumerate(series_file_names):
            #open file to get sop id
            ds = pydicom.dcmread(loccPath)
            sop=mainFuncs.get_SOPInstanceUID(ds)
            annot_for_sop= annot_for_label.loc[annot_for_label['SOPInstanceUID'] == sop]       
            if(len(annot_for_sop)>0):
                # print(f"2 lllllll lab {lab} ")

                rowOfIntr=list(annot_for_sop.iterrows())[0]
                #we obtain mask as a boolean array
                binArray= mainFuncs.load_mask_instance(rowOfIntr).astype(dtype)

                #time to overwrite the data
                # ds.PixelData = binArray.tostring()
                if((data.shape[1],data.shape[2])==binArray.shape):
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
        zeroArray= add_layer_to_be_consistent(zeroArray)

        image = sitk.GetImageFromArray(zeroArray.astype(np.uint16))  
        image.SetSpacing(image3D.GetSpacing())
        image.SetOrigin(image3D.GetOrigin())
        try:
            image.SetDirection(image3D.GetDirection())    
        except:
            print(f"error saving direction of {labelNiiPath}")
            
        writer.SetFileName(labelNiiPath)
        writer.Execute(image)
        labelSum=np.sum(zeroArray)
        if(labelSum>0):
            save_dicom_seg_label([],jsonFolder,' ',outPath=labelSegPath,dicomSPath=dicomSPath,seriesId=seriesId,segmentation=image,curr_no_space_name=labb)
        # print(f"gggg getLabelsAbbrev(lab) {getLabelsAbbrev(lab)},labelNiiPath {labelNiiPath} labelSum {labelSum} ")
    return (getLabelsAbbrev(lab),labelNiiPath,labelSegPath,labelSum)

def translateSeriesDesc(series_desc_string,acqNumb):
    """ 14*36
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
        ,("'ep2d_diff_b 50 400 800 calc 1200_ADC'"               ,"adc")    
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
        ,("'t2_bl_SAG_tra_P'"                               ,"sag")
        
        ]

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
    # locDf = files_df.loc[files_df['SeriesInstanceUID'] == currentSeries]
    locDfB = files_df_origFolds.loc[files_df_origFolds['SeriesInstanceUID'] == currentSeries]
    # locDf=locDf.drop_duplicates(subset = ["SOPInstanceUID"])
    # in order to maximise the chance that we will getaround tag problems we will use dicoms from both sources
    # paths_in_series= np.unique(locDf['paths'].to_numpy())
    paths_in_series= np.unique(locDfB['paths'].to_numpy())
    # paths_in_series=[*paths_in_series,*paths_in_seriesB]
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

def saveMainMRI(paths_in_series,paths_dict,currentStudyDesc,currentSeries,masterolds):
    """
    given paths to  dicom files with unique slices we will copy those into separate dicom folder and 
    generate a single mha file 
    """
    origVolPath=join(paths_dict['data_path_seg'],f"{masterolds}_{currentStudyDesc}")
    os.makedirs(origVolPath ,exist_ok = True)
    pathMha=join(paths_dict['data_path_no_seg'],f"{masterolds}_{currentStudyDesc}.mha")
    
    image3D=[]
    series_file_names=[]
    if(not pathOs.exists(pathMha)):
        try:
            for path_to_copy in paths_in_series:
                dicomResPath=join(origVolPath,Path(path_to_copy).name)
                shutil.copyfile(path_to_copy,dicomResPath)
                ds = pydicom.read_file(dicomResPath)
                #(0008,1030) StudyDescription TEST
                # AI4AR_PCa
                ds[0x0008, 0x1030].value = "AI4AR_PCa"

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
            return origVolPath,pathMha,' ',' ',[]
    else:
        #no point in recreating if it already exist
        image3D=sitk.ReadImage(pathMha)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(origVolPath,currentSeries)

    return origVolPath,pathMha,sitk.GetArrayFromImage(image3D),image3D,series_file_names




def mainGenereteFiles(files_df,files_df_origFolds,annot_for_series,files_for_series,currentSeries,current_study_id,mainPaths_studyId,masterolds_in_Study
,jsonFolder,corrections_for_study_id,pureMasterNum):
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
        origVolPath,pathMha,image3Ddata,image3D,series_file_names = saveMainMRI(paths_in_series,paths_dict,currentStudyDesc,currentSeries,masterolds_in_Study)
        # in case of error in the main image reading we will not return anything
        if(len(series_file_names)==0):
            return (' ',' ',' ',' ',' ',' ' ,' ' )    
        
        ## saving anatomical labels (if they exist)
        #we will check how many diffrent labels are associated in case of no annotations it will be empty
        uniq_labels= np.unique(annot_for_series['labelName'].to_numpy())

        # print(f"uuuuuuuuuuuniq_labels {uniq_labels}")


        uniq_labels= list(filter(lambda labb: labb!='lymph node regional group',uniq_labels  ))
        uniq_labels= list(filter(lambda labb: labb!='curvilinear contact',uniq_labels  ))
        uniq_labels= list(filter(lambda labb: labb!='lymph node regional',uniq_labels  ))
        uniqLabelsAnatomy = list(filter(lambda lab: 'lesion' not in lab,uniq_labels))
        labelNameAndPaths=[]
        if(len(uniqLabelsAnatomy)>0):
            labelNameAndPaths=list(map(lambda lab: createAnatLabel(annot_for_series,lab,image3Ddata,paths_dict
            ,series_file_names,image3D,masterolds,currentStudyDesc,jsonFolder,origVolPath,currentSeries),uniqLabelsAnatomy ))
        labelNameAndPaths=list(filter(lambda el: el[0]!=' ',labelNameAndPaths))
        # saving data about lesions separately for each doctor (annotator)
        for docId in np.unique(annot_for_series['createdById'].to_numpy()):
            annot_for_doctor=annot_for_series.loc[annot_for_series['createdById'] == docId]  
            corrections_for_doc= corrections_for_study_id.loc[corrections_for_study_id['rad_hash'] == docId]    
            # getting all the labels pesent for this series and this annotator                
            uniq_labels= np.unique(annot_for_doctor['labelName'].to_numpy())
            uniqLabelsLesions = list(filter(lambda lab: 'lesion' in lab,uniq_labels))
            for lab in uniqLabelsLesions:
                res = createLesionLabel(annot_for_doctor,lab,image3Ddata,paths_dict,series_file_names,image3D,masterolds
                ,docId,currentStudyDesc,jsonFolder,origVolPath,currentSeries,corrections_for_doc )
                labelNameAndPaths.append(res)

            uniq_new_anatomy = list(filter(lambda abb: ((abb=='bladder lumen') or (abb=='bladder wall') or (abb=='rectus abdominis muscle L') or (abb=='rectus abdominis muscle R')) ,uniq_labels))
            if(len(uniq_new_anatomy)>0):
                print(f"uuuuuniq_new_anatomy {uniq_new_anatomy}")
            for lab in uniq_new_anatomy:
                res = createNew_anatomy_label(annot_for_doctor,lab,image3Ddata,paths_dict,series_file_names,image3D,masterolds
                ,docId,currentStudyDesc,jsonFolder,origVolPath,currentSeries,corrections_for_doc )
                labelNameAndPaths.append(res)

        #filter out all errors and labels that should be removed        
        labelNameAndPaths=tuple(list(filter(lambda el: el[0]!=' ',labelNameAndPaths)))
        return (current_study_id,currentSeries,currentStudyDesc,pathMha,origVolPath,labelNameAndPaths ,pureMasterNum )
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

def iterate_overStudy(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder,correctionsFrame,neededIds):
    """
    iterate ove all series with the same study UID
    """
    res=[]
    # print(f"current_study_id {current_study_id}")
    annot_for_study_id=annot.loc[annot['StudyInstanceUID'] == current_study_id]
    files_for_study_id=files_df_origFolds.loc[files_df_origFolds['StudyInstanceUID'] == current_study_id]


    masterolds_in_Study=files_for_study_id['masterolds'].to_numpy()

    masterolds=str(masterolds_in_Study[0])#.replace('nas-lssi-dco/','')
    #checking weather we are intrested in a file at all
    pureMasterNum=str(int(masterolds))
    patIdsCorrs=correctionsFrame['patient_id'].to_numpy()

    corrections_for_study_id=correctionsFrame.loc[correctionsFrame['patient_id'] ==int(pureMasterNum) ]

    annot_for_study_id_print = annot_for_study_id[annot_for_study_id['labelName'].str.contains('lesion')]

    # print(f"pureMasterNum {pureMasterNum} \n corrections_for_study_id \n {corrections_for_study_id} \n annot_for_study_id \n {annot_for_study_id_print[['createdById', 'labelName']]}")
    if(masterolds==' '):
        masterolds=f"unknownMasterNum_{current_study_id}"
        print("unknownnn masterrr ")
    mainPaths_studyId=list(map(partial(createStudyFolder,masterolds=masterolds),mainPaths.items()))

    for currentSeries in np.unique(files_for_study_id['SeriesInstanceUID'].to_numpy()):
        annot_for_series=annot_for_study_id.loc[annot_for_study_id['SeriesInstanceUID'] == currentSeries]
        files_for_series=files_for_study_id.loc[files_for_study_id['SeriesInstanceUID'] == currentSeries]

        res.append(mainGenereteFiles(files_df,files_df_origFolds,annot_for_series,files_for_series
            ,currentSeries,current_study_id,mainPaths_studyId,masterolds,jsonFolder,corrections_for_study_id,pureMasterNum))
    # for el in res:
    #     if(not (type(el) is str)):
    #         # if(not (type(el) is tuple)):

    #             print(type(el))
    
    # res= np.array(res)
    res=list(filter( lambda el: el[0]!=' ' ,res))     

    return res
    # #if we are not intrested in the file return dummy
    # print(f"aaa filtered out {masterolds}")
    # return []

def iterate_overStudySafe(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder,correctionsFrame,neededIds):
    """
    just gives one restart of iterate_overStudy function in case of failure
    """
    try:
        return iterate_overStudy(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder,correctionsFrame,neededIds)
    except:
        return iterate_overStudy(current_study_id,files_df,files_df_origFolds,annot,outputDir,mainPaths,jsonFolder,correctionsFrame,neededIds)


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
    # print(f"getLabelNames {res}")
    return res




def get_frame_with_output(files_df,annot,outputDir,resCSVDir,mainFoldDirMha,mainFoldDirSeg,jsonFolder,correctionsCSVDir,neededNumbersCSVDir):
    """
    in parallel iterates over all studies and series and save the paths of created files in the csv file
    """
    
    if(pathOs.exists(resCSVDir)):
        return pd.read_csv(resCSVDir) 
    

    out_files_frame= pd.DataFrame()
    correctionsFrame=pd.read_csv(correctionsCSVDir)
    neededNumbersCSV=pd.read_csv(neededNumbersCSVDir)
    neededIds= np.unique(neededNumbersCSV['patient_id'].to_numpy())
    neededIds = list(map( lambda el: int(el),neededIds))

    #filtering only rows that we are intrested in 
    yy = list(files_df['masterolds'].to_numpy())
    # print(f"yyyyyy {yy}")
    # yy= list(map(lambda el: int("".join(list(filter(lambda l: l.isnumeric(),el)))) in neededIds,yy))
    yy= list(map(lambda el: el in neededIds,yy))

    # files_df=files_df.loc[yy]

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
        allPaths=pool.map(partial(iterate_overStudySafe, files_df=files_df,files_df_origFolds=files_df,annot=annot,outputDir=outputDir,mainPaths=mainPaths,jsonFolder=jsonFolder,correctionsFrame=correctionsFrame,neededIds=neededIds), np.unique(files_df['StudyInstanceUID'].to_numpy()))
    
    # allPaths=list(map(partial(iterate_overStudySafe, files_df=files_df,files_df_origFolds=files_df_origFolds,annot=annot,outputDir=outputDir,mainPaths=mainPaths,jsonFolder=jsonFolder,correctionsFrame=correctionsFrame,neededIds=neededIds), np.unique(files_df_origFolds['StudyInstanceUID'].to_numpy())))    #filtering out all cases where we returned a dummy
    # allPaths= list(filter(lambda el: len(el)>0,allPaths))

    flatten_list_paths = list(itertools.chain(*allPaths))
    #we filter out all of the cases that were excluded becouse for example they were incomplete ...
    # flatten_list_paths= list(filter(lambda tupl: int(tupl[6]) in  neededIds ,flatten_list_paths))

    out_files_frame['study_id']=list(map(lambda tupl: tupl[0],flatten_list_paths))
    # out_files_frame['doctor_id']=list(map(lambda tupl: tupl[1],flatten_list_paths))
    out_files_frame['series_id']=list(map(lambda tupl: tupl[1],flatten_list_paths))
    out_files_frame['series_desc']=list(map(lambda tupl: tupl[2],flatten_list_paths))
    out_files_frame['series_MRI_path']=list(map(lambda tupl: tupl[3],flatten_list_paths))
    out_files_frame['dicom_MRI_path']=list(map(lambda tupl: tupl[4],flatten_list_paths))
    out_files_frame['masterolds']=list(map(lambda tupl: tupl[6],flatten_list_paths))
    out_files_frame=out_files_frame.copy()
    all_labels_types=list(map(lambda el: getLabelNames(el),flatten_list_paths))
    all_labels_types=list(itertools.chain(*all_labels_types))
    all_labels_types=np.unique(all_labels_types)

    for targetLab in all_labels_types :
        out_files_frame=out_files_frame.copy()
        out_files_frame[f"{targetLab}_noSeg"]=list(map(lambda tupl: getLabelPathOrEmpty(targetLab,tupl,1),flatten_list_paths))
        out_files_frame[f"{targetLab}_Seg"]=list(map(lambda tupl: getLabelPathOrEmpty(targetLab,tupl,2),flatten_list_paths))
        out_files_frame[f"{targetLab}_num"]=list(map(lambda tupl: getLabelPathOrEmpty(targetLab,tupl,3),flatten_list_paths))

    out_files_frame.to_csv(resCSVDir) 
    print(f"out files frame ready")
    return out_files_frame




# out_files_frame.columns

# len(allPaths[0])


# flatten_list[0][4]

# allPaths[0][0][0]

# allPaths


# ## get single study
# for current_study_id in np.unique(annot['StudyInstanceUID'].to_numpy()):
    



#//home/sliceruser/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.8.15.100960.30000022021714130657000003016/U_OjmNZ8/1.3.12.2.1107.5.8.15.100960.30000022021714130657000003047/volume.mha
#//home/sliceruser/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.8.15.100960.30000022021714463775000001099/U_OjmNZ8/1.3.12.2.1107.5.8.15.100960.30000022021714463775000001281/volume.mha

#//home/sliceruser/workspaces/konwersjaJsonData/output/1.3.12.2.1107.5.8.15.100960.30000022021714130657000000014/U_8ZmM76/1.3.12.2.1107.5.8.15.100960.30000022021714130657000001555/volume.mha




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