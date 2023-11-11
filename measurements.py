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
import time
import mainFuncs
import itertools
import torch
from os import path as pathOs
import more_itertools
from os.path import basename, dirname, exists, isdir, join, split
import shutil
import glob
import os

import numpy as np
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
from itertools import product



def save_from_arr(zeroArray,image3D,newPathLab):
    """
    given array saves it to file into defined path using simpleitk
    """
    writer = sitk.ImageFileWriter()
    image = sitk.GetImageFromArray(zeroArray)  
    image.SetSpacing(image3D.GetSpacing())
    image.SetOrigin(image3D.GetOrigin())
    image.SetDirection(image3D.GetDirection())    

    writer.SetFileName(newPathLab)
    writer.Execute(image)


def get_volume(path):
    """
    get volume of a binary mask that resides in path
    """
    image3D=sitk.ReadImage(path)
    data=sitk.GetArrayFromImage(image3D)
    spacing = image3D.GetSpacing()
    #we get total vlume by multiplying sum of all voxels with each voxel volume
    volume = np.sum(data)*(spacing[0]*spacing[1]*spacing[2])/1000
    return volume

def get_dice_between(pathA,pathB):
    """
    from http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    get two images and return dice coefficient
    """
    imageA=sitk.ReadImage(pathA)
    imageB=sitk.ReadImage(pathB)
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(imageA, imageB)
    return overlap_measures_filter.GetDiceCoefficient()




def getVolumes(current_row,col_names_to_volume):
    """
    given data of single series return the volume of each label 
    in case given label is not present return -1
    """
    current_row= current_row[1]
    #getting valid column names for this row
    col_names_for_volumes_inner = list(filter(  lambda colName :current_row[colName]!=" " , col_names_to_volume)) 
    rest_of_colnames= list(filter(  lambda colName :current_row[colName]==" " , col_names_to_volume))
    #getting Paths
    volumesPaths = list(map(lambda colName:current_row[colName],col_names_for_volumes_inner))
    #calculate volumes of each label
    volumes= list(map(get_volume, volumesPaths))
    #dummy tensor with minus 1 to keepshape consistent
    volumes_rest=  list(map(lambda x:-1 ,rest_of_colnames))
    return list(zip(col_names_for_volumes_inner+rest_of_colnames,volumes+volumes_rest))



def get_label_per_doc(doctor_id,locDf,col_names_for_dice):
    """
    utility function getting label for each annotator and combines it into single string
    """
    locLocDf = list(locDf.loc[locDf['doctor_id'] == doctor_id].iterrows())[0][1]
    #we filter all lesion names that happen to be present with this annotator doctor
    liist= list( filter(lambda clN: locLocDf[clN]!=" " ,col_names_for_dice))
    liist= list( map(lambda clN: clN+'_@_'+locLocDf[clN] ,liist))
    #we will add also a path to the lesions of intrest
    return list(map(lambda entry : doctor_id+'_@_'+entry   ,liist))



def get_dice_from_tupl(tupl, locDf):
    """
    given tuple holding data about label and annotator return the associated DICE score
    """
    changeA= tupl[0].split('_@_')
    changeB= tupl[1].split('_@_')
    
    doctorA= changeA[0]
    doctorB= changeB[0]

    labelA = changeA[1]
    labelB = changeB[1]

    pathA = list(locDf.loc[locDf['doctor_id'] == doctorA].iterrows())[0][1][labelA]
    pathB = list(locDf.loc[locDf['doctor_id'] == doctorB].iterrows())[0][1][labelB]
    return get_dice_between(pathA,pathB)

def splitLabelDataForFrame(tupl):
    """
    given tupl with data return list that is ready to be row in the output dataframe
    """
    return(tupl[0][0]#series id
    ,tupl[0][1].split('_@_')[0] #doctor a
    ,tupl[0][1].split('_@_')[1] #lesion a
    ,tupl[0][1].split('_@_')[2] #path to lesion a
    ,tupl[0][2].split('_@_')[0] #doctor b
    ,tupl[0][2].split('_@_')[1] #lesion b
    ,tupl[0][2].split('_@_')[2] #path to lesion b
    ,tupl[1]#dice
    ,tupl[2]  #mriPath    
    ) 

def get_dice_in_all_pairs(currentSeries,preprocessed_df,col_names_for_dice):
    """
    analyzing all possible combination o lesions descriped by diffrent annotators
    calculating dice score between them and returning nested tuple with series id and dice score for each pair
    """
    
    locDf = preprocessed_df.loc[preprocessed_df['series_id'] == currentSeries]
    doctors= np.unique(locDf['doctor_id'].to_numpy())
    mriPath= np.unique(locDf['series_MRI_path'].to_numpy())[0]

    if(len(doctors)>1):
        #getting the combinations of possible labels and doctor comparing all
        labels_per_doctor =list( map(partial(get_label_per_doc, col_names_for_dice=col_names_for_dice,locDf=locDf ),doctors   ))
        labels_per_doctor=list(itertools.chain(*labels_per_doctor))


        labels_per_doctor_all_pairs=list(more_itertools.powerset(labels_per_doctor))
        #filter only those of length 2
        labels_per_doctor_all_pairs=list(filter(lambda tupl:len(tupl)==2  ,labels_per_doctor_all_pairs))
        #filter only those that has in single tuple diffrent doctors
        labels_per_doctor_all_pairs=list(filter(lambda tupl: tupl[0].split('_@_')[0]!=tupl[1].split('_@_')[0]  ,labels_per_doctor_all_pairs))
        #calculating dice values
        dice_vals= list(map(partial(get_dice_from_tupl,locDf=locDf), labels_per_doctor_all_pairs ))
        #adding data about series to each tuple
        labels_per_doctor_all_pairs= list(map( lambda tupl: (currentSeries,tupl[0],tupl[1] )   ,labels_per_doctor_all_pairs))
        labels_per_doctor_all_pairs= list(zip(labels_per_doctor_all_pairs,dice_vals))
        labels_per_doctor_all_pairs= list(map(lambda tupl: (tupl[0],tupl[1],mriPath  ) ,labels_per_doctor_all_pairs))

        
        #prepared for returning
        labels_per_doctor_all_pairs=list(map(splitLabelDataForFrame   ,labels_per_doctor_all_pairs))
        return labels_per_doctor_all_pairs
    return [(" ", " ", " ", " ", " ", " ", " ", " "," ")]

def get_volume_and_dice_data(col_names_for_dice,col_names_to_volume, preprocessed_df,volumes_csv_dir,dice_csv_dir):
    """
    reads the data from associated data frame and creates new dataframe
    with data about volumes 
    with data about the volume of each label where we want it
    additionally creates a new dataframe with data about dice scores
    """
    all_volumes_data= list(map( partial(getVolumes,col_names_to_volume=col_names_to_volume)   , preprocessed_df.iterrows()))
    volumes_frame= pd.DataFrame()
    #populating frame with data
    volumes_frame['study_id']=preprocessed_df['study_id']   
    volumes_frame['doctor_id']=preprocessed_df['doctor_id']   
    volumes_frame['series_id']=preprocessed_df['series_id']
    
    for col_vol_name in col_names_to_volume:
        curr_vol_dat= list(map(lambda zipped : list(filter(lambda tupl: tupl[0]==col_vol_name ,zipped ))[0]  ,all_volumes_data))
        volumes_frame[col_vol_name]=curr_vol_dat
    
    #### Dice
    series = np.unique(preprocessed_df['series_id'].to_numpy())
    list_dice_score = list(map( partial(get_dice_in_all_pairs,preprocessed_df=preprocessed_df,col_names_for_dice=col_names_for_dice),series ))
    list_dice_score=list(itertools.chain(*list_dice_score))

    #so now we have the series of nested tuples entry 1 is series uid second entry is list of tuples where fitst entry marks what
    #lesion and which annotator did it
    # goal isto  create dataframe where column
        #1) series id 2) first annotator id 3) first lesion name 4) second annotator id 5) socond lesion name 6) dice score
    dice_df = pd.DataFrame(list_dice_score, columns =['SeriesId', 'doctor_a', 'lesion_a','path_lesion_a','doctor_b','lesion_b','path_lesion_b', 'dice','mriPath'])
    volumes_frame.to_csv(volumes_csv_dir) 
    dice_df.to_csv(dice_csv_dir)
    return (all_volumes_data,dice_df) 


def labelNamesForDoc(doc,locDf):
    """
    given dataframe it looks for all of the names of lesions associated with given human annotator
    """
    rows = list(locDf.iterrows())
    rows = list(map(lambda row:row[1],rows))
    rowsA = list(filter(lambda row:row['doctor_a']==doc,rows))
    rowsB = list(filter(lambda row:row['doctor_b']==doc,rows))
    rowsA=list(map(lambda row: row['lesion_a'] ,rowsA   ))
    rowsB=list(map(lambda row: row['lesion_b'] ,rowsB   ))
    concated= rowsA+rowsB#+rowsB #np.concatenate([ rowsA,rowsB ])

    #print(f"rowsB {rowsB}   plus {concated}   ")
    # return np.unique(list(itertools.chain(*concated)))
    return np.unique(concated)


def getLenInDoc(tupl):
    """
    helper function to find doctor with biggest number of lesions described
    """
    return len(tupl[1])

def getmaxDiceInDoc(row):
    """
    helper function to find doctor with biggest number of lesions described
    """
    return row['dice']

def choosePath(row, docName):
    """
    choose the path associated with human annotator
    """
    if(row['doctor_a']==docName):
        return row['path_lesion_a']
    return row['path_lesion_b'] 

def chooseLesionName(row, docName):
    """
    choose the path associated with human annotator
    """
    if(row['doctor_a']==docName):
        return row['lesion_a']
    return row['lesion_b'] 


def getForSaveName(row, docName):
    """
    creates a string for creating file name
    """
    if(row['doctor_a']==docName):
        return row['lesion_a']+'_'+docName
    return row['lesion_b']+'_'+docName


def save_lesions_consensus(dice_df,rootFolder_lesion_analysis ):
    """
    on the basis of the data from dice frame we will establish the consensus of the images and save them to appropriate set folder
    """
    #we have list of SeriesId,doctor_a,lesion_a,path_lesion_a,doctor_b,lesion_b,path_lesion_b,dice
    #fisrt we will filter rows where dice is not 0
    preprocessed_df=dice_df.loc[dice_df['dice'] != " "]
    #preprocessed_df=preprocessed_df.loc[preprocessed_df['dice'] != 0]
    #then we get all unique series id
    series = np.unique(preprocessed_df['SeriesId'].to_numpy())
    res=list(map(lambda currentSeries : iter_over_series_and_save_cons(currentSeries,preprocessed_df,rootFolder_lesion_analysis ),series ))

def iter_over_series_and_save_cons(currentSeries,preprocessed_df,rootFolder_lesion_analysis ):
    """
    iterates over each image series and saves the consensus
    """
    debugSeries='1.3.12.2.1107.5.2.41.69644.202006090804423911430615.0.0.0'
    if(currentSeries==debugSeries):
        ##per unique series id
        locDf = preprocessed_df.loc[preprocessed_df['SeriesId'] == currentSeries]

        #create folder with this series id and copy there the MRI mha file and original lesion files
        locFolderPath=join(rootFolder_lesion_analysis,currentSeries)
        os.makedirs(locFolderPath,exist_ok=True)
        #get some mri path it should point to the same mri in all cases for this series
        mriPath=list(locDf['mriPath'].to_numpy())[0]
        #then  unique lesion names and doctor names from lesion_a and lesion_b in this series n_lesions
        lesion_names= np.unique(np.concatenate([locDf['lesion_a'].to_numpy(),locDf['lesion_b'].to_numpy()]))
        doctor_names= np.unique(np.concatenate([locDf['doctor_a'].to_numpy(),locDf['doctor_b'].to_numpy()]))
        #**copy files
        aZipped=list(zip(locDf['lesion_a'].to_numpy(),locDf['doctor_a'].to_numpy(),locDf['path_lesion_a'].to_numpy()    ))
        bZipped=list(zip(locDf['lesion_b'].to_numpy(),locDf['doctor_b'].to_numpy(),locDf['path_lesion_b'].to_numpy()    ))
        allZipped= list(np.concatenate([aZipped,bZipped]))
        #get only unique
        allZipped=list(map(lambda tupl: tupl[0]+'___' +tupl[1]+'___'+tupl[2], allZipped  ))
        allZipped=np.unique(allZipped )
        allZipped=list(map( lambda fused : fused.split("___")  ,allZipped ))
        #name of the copied file will point to the lesion name and annotator
        list(map( lambda tupl: shutil.copyfile(tupl[2] ,join( locFolderPath ,tupl[0]+'_'+tupl[1]+'.nii.gz')  ),allZipped))
        shutil.copyfile(mriPath, join(locFolderPath, 'volume.mha'))

        #we select the doctor with most lesion names present
        doctor_lesions=list(map(partial(labelNamesForDoc,locDf=locDf) ,doctor_names))
        print(f"doctor_lesions {doctor_lesions}")
        zipped= list(zip(doctor_names, doctor_lesions))
        #data about human annotator with max number of lesions described
        docTuplMax=max(zipped, key=getLenInDoc)
        print(f"docTuplMax {docTuplMax}")
        #now we keep doc max as reference we iterate over his/her lesions and associate it with lesions of other annotator with highest dice score with this lesion
        maxDocName=docTuplMax[0]
        lesions_to_analyze=docTuplMax[1]

        doctors_not_max=list(filter(lambda docName:docName!=maxDocName  ,doctor_names))
        
        #print(f"doctors_not_max {doctors_not_max} all docs {doctor_names}")

        rows=list(locDf.iterrows())
        rows=list(map(lambda row: row[1] ,rows))

        list(map( lambda current_lesion : alalyze_per_lesion_for_consensus(current_lesion,maxDocName,doctors_not_max,mriPath,locFolderPath,rows )  ,lesions_to_analyze))
        # with mp.Pool(processes = mp.cpu_count()) as pool:
        #     pool.map( lambda current_lesion : alalyze_per_lesion_for_consensus(current_lesion,maxDocName,doctors_not_max,mriPath,locFolderPath,rows )  ,lesions_to_analyze)



def alalyze_per_lesion_for_consensus(current_lesion,maxDocName,doctors_not_max,mriPath,locFolderPath ,rowsOut):
    #we choose sinlge lesion and get rows only associated with it
    rows=list(filter(lambda row: 
                            (row['lesion_a']==current_lesion and row['doctor_a']==maxDocName)
                            or 
                            (row['lesion_b']==current_lesion and row['doctor_b']==maxDocName ),rowsOut ))
    #now we will analyze non chosen doctors and from their data we will choose the image that is most simmilar to chosen label of max doctor
    perDocRows=list(map(lambda doctorNotMaxName :list(filter(lambda row: 
                            (row['doctor_a']==doctorNotMaxName or row['doctor_b']==doctorNotMaxName)
                            ,rows )) ,doctors_not_max))

    toPrint=list(map(lambda listt: len(listt)  ,perDocRows))
    
    # print(f"perDocRows len {toPrint} rows len {len(rows)} ")
    # print(f"lesionA {list(map(lambda row:row['lesion_a']   ,rowsOut))}   ")
    # print(f"lesionB {list(map(lambda row:row['lesion_b']   ,rowsOut))}   ")
    # print(f"doctor_a {list(map(lambda row:row['doctor_a']   ,rowsOut))}   ")
    # print(f"doctor_b {list(map(lambda row:row['doctor_b']   ,rowsOut))}   ")

    if(len(rows)>0  ):
        perDocRows=list(filter(lambda docRows : len(docRows)>0   ,perDocRows))
        perDocRows=list(map(lambda docRows : max(docRows, key=getmaxDiceInDoc)   ,perDocRows))

        zipped_perDocRows= list(zip(doctors_not_max,perDocRows ))

        #establishing what needs to be fused to obtain consensus
        paths_to_fuse= list(map( lambda tupl: choosePath(tupl[1],tupl[0]) ,zipped_perDocRows))
        images_toFuse= list(map( sitk.ReadImage ,paths_to_fuse))
        images_toFuse= list(map( sitk.GetArrayFromImage ,images_toFuse))
        images_toFuse= list(map(lambda imDat : imDat.astype(bool) ,images_toFuse))
        fused = functools.reduce(np.logical_and, images_toFuse)
        name_not_main=list(map(lambda tupl: getForSaveName(tupl[1],tupl[0])  ,zipped_perDocRows))
        name_not_main='_'.join(name_not_main)
        name=current_lesion+'_'+maxDocName+'_'+name_not_main
        name=name.replace(' ','_')
        name=join(locFolderPath,name+'.nii.gz')
        #original MRI to load metadata
        image3D=sitk.ReadImage(mriPath)
        #saving the consensus of 3 images
        save_from_arr(fused.astype(np.uint8),image3D,name)
        #saving the paired consensus
        #first getting image data
        biPathImage=list(map(lambda row : (row['path_lesion_a'], row['path_lesion_b']   )  ,perDocRows   ))
        biPathImage=list(map(lambda paths : ( sitk.ReadImage(paths[0]) , sitk.ReadImage(paths[1]) )  ,biPathImage   ))
        biPathImage=list(map(lambda images :( sitk.GetArrayFromImage(images[0]) , sitk.GetArrayFromImage(images[1]) )  ,biPathImage   ))
        fusedBiImages=list(map(lambda imagesDat : np.logical_and( imagesDat[0].astype(bool),imagesDat[1].astype(bool))  ,biPathImage   ))
        #now get name for the new file with bi consensus
        fusedBiImageNames= list(map( lambda row:row["doctor_a"]+'_'+row["lesion_a"]+'_'+row["doctor_b"]+'_'+row["lesion_b"]+'.nii.gz'  ,perDocRows ))
        fusedBiImageNames= list(map( lambda nameIn:join(locFolderPath,nameIn)  ,fusedBiImageNames ))
        fusedBiImageNames= list(map( lambda nameIn:nameIn.replace(' ','_')  ,fusedBiImageNames ))
        zippedBiImage=list(zip(fusedBiImages,fusedBiImageNames   ))
        #saving the paired consensus
        list(map( lambda tupl : save_from_arr(tupl[0].astype(np.uint8),image3D,tupl[1]) ,zippedBiImage  ))
        #return paths of the saved files
        return [*fusedBiImageNames,name ]


def is_newanatomy_in(row,new_anatomy_cols):
    is_sth=list(map(lambda name : len(row[name])>2, new_anatomy_cols))
    return np.sum(is_sth)>0

def get_annotator_id(path,path_info ):
    """ 
    based on path return annotator id
    """
    # print(f"in det annot id filterr {list(filter(lambda tupl: tupl[0]==path ,path_info))} path {path}")
    col_name=list(filter(lambda tupl: tupl[0]==path ,path_info))[0][1]
    return f"{col_name.split('_')[-4]}_{col_name.split('_')[-3]}"

def analyze_pair(pair_paths,path_info,name,evaluator,patient_id):
    namee=f"{patient_id}_|_{name}_|_{get_annotator_id(pair_paths[0],path_info )}_|_{get_annotator_id(pair_paths[1],path_info )}"
    evaluator.evaluate(sitk.ReadImage(pair_paths[0]), sitk.ReadImage(pair_paths[1]), namee)

def analyze_same_anatomy(name,row,new_anatomy_cols,evaluator):
    """ 
    first we find all of the columns that has appropriate name
    then we collect the files and compare the using dice and hausdorff
    return info about what is compared and what is the score and what metric used
    """
    colss= list(filter( lambda col_name: name in col_name,new_anatomy_cols))
    pathss = list(map(lambda col_name :row[col_name] ,colss ))
    path_info=list(zip(pathss,colss))
    path_info=list(filter(lambda tupl: 'konwersjaJsonData' in tupl[0] ,path_info))
    pathss= list(map(lambda tupl: tupl[0],path_info))

    # print(f"pppp pathss {pathss}")

    cart_prod=list(product(pathss,pathss))
    cart_prod= list(filter(lambda tupl: tupl[0]!=tupl[1],cart_prod))
    
    list(map(lambda pair_paths :analyze_pair(pair_paths,path_info,name,evaluator,row['masterolds']),cart_prod))
    





# writer.CSVWriter(result_file).write(evaluator.results)
# functions = {'MEAN': np.mean, 'STD': np.std}
# writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
# print('\nAggregated statistic results...')
# writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

def analyze_row(row,new_anatomy_cols,evaluator):
    """ 
    we have row where some new anatomy was analyzed
    """
    names= ['bladder_lumen','bladder_wall','rec_abd_L','rec_abd_R']
    list(map(lambda name :analyze_same_anatomy(name,row,new_anatomy_cols,evaluator),names))

def get_new_anatomu_inter_observer_agreement(preprocessed_df,new_anatomy_csv_dir):
    """ 
    looking for abdomen and bladder annotations and noting how much in agreement the annottors were
    """
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'), metric.VolumeSimilarity()]
    labels = {1: 'segmentation' }
    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    cols=preprocessed_df.columns
    cols=list(filter(lambda el: 'bladder_lumen' in el or 'bladder_wall' in el or 'rec_abd_L' in el or 'rec_abd_R' in el ,cols))
    new_anatomy_cols=list(filter(lambda el: '_noSeg' in el ,cols))
    rows= preprocessed_df.iterrows()
    rows= list(map(lambda el: el[1],rows))
    rows= list(filter(lambda row :is_newanatomy_in(row,new_anatomy_cols) ,rows))

    list(map(lambda row : analyze_row(row,new_anatomy_cols,evaluator),rows))
    writer.CSVWriter(new_anatomy_csv_dir).write(evaluator.results)
    frame = pd.read_csv(new_anatomy_csv_dir,header=0,sep=";")

    print(f"ffff {frame.columns}")
    subj=frame['SUBJECT'].to_numpy()
    subj = list(map(lambda el: el.split('_|_'),subj ))
    frame["patient_id"]=list(map(lambda el: el[0]  ,subj))
    frame["organ"]=list(map(lambda el: el[1]  ,subj))
    frame["annotator_a"]=list(map(lambda el: el[2]  ,subj))
    frame["annotator_b"]=list(map(lambda el: el[3]  ,subj))

    rafal_id='U_8nPj9Q'
    frame.loc[frame['annotator_a'] != rafal_id ]
    frame.loc[frame['annotator_b'] != rafal_id ]


    frame.to_csv(new_anatomy_csv_dir)


#/media/jakub/NewVolume/projects/konwersjaJsonData/forLesionAnalysis/1.3.12.2.1107.5.2.41.69644.202006090804423911430615.0.0.0
#1.3.12.2.1107.5.2.41.69644.202006090804423911430615.0.0.0
