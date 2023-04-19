import SimpleITK as sitk
from subprocess import Popen
import subprocess
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import functools
from functools import partial
import sys
import os.path
from os import path as pathOs
import numpy as np
import tempfile
import shutil
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
import fileinput
import re
import subprocess


def transform_label(path_label,out_folder,transformix_path ,transformixParameters):
    outPath_label= join(out_folder,Path(path_label).name.replace(".nii.gz",""))
    os.makedirs(outPath_label ,exist_ok = True)
    cmd_transFormix=f"{transformix_path} -in {path_label} -def all -out {outPath_label} -tp {transformixParameters} -threads 1"
    p = Popen(cmd_transFormix, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)
    p.wait()
    return join(outPath_label,'result.mha')


def reg_a_to_b(out_folder,patId,path_a,path_b,labels_b_list,reg_prop ,elacticPath,transformix_path,modality,reIndex=0):
    """
    register image in path_a to image in path_b
    then using the same registration procedure will move all of the labels associated with path_b to the same space
    as path_a
    out_folder - folder where results will be written
    elactic_path- path to elastix application
    transformix_path  = path to transformix application
    reg_prop - path to file with registration

    return a tuple where first entry is a registered MRI and second one are registered labels
    """
    path=path_b
    outPath = out_folder
    os.makedirs(out_folder ,exist_ok = True)    
    result=pathOs.join(outPath,"result.0.mha")


    cmd=f"{elacticPath} -f {path_a} -m {path} -out {outPath} -p {reg_prop} -threads 1"
    p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
    p.wait()
    #we will repeat operation multiple max 9 times if the result would not be written
    if((not pathOs.exists(result)) and reIndex<3):
       
        reg_prop=reg_prop.replace("parameters","parametersB")

        cmd=f"{elacticPath} -f {path_a} -m {path} -out {outPath} -p {reg_prop} -threads 1"

        p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
        p.wait()

        # reIndexNew=reIndex+1
        # if(reIndex==1): #in case it do not work we will try diffrent parametrization
        #     reg_prop=reg_prop.replace("parameters","parametersB")              
        # #recursively invoke function multiple times in order to maximize the probability of success    
        # reg_a_to_b(out_folder,patId,path_a,path_b,labels_b_list,reg_prop ,elacticPath,transformix_path,modality,reIndexNew)
    if(not pathOs.exists(result)):
        print("registration unsuccessfull")
        return " "
    print("registration success")
    transformixParameters= join(outPath,"TransformParameters.0.txt")
    # they can be also raw string and regex
    textToSearch = 'FinalBSplineInterpolator' # here an example with a regex
    textToReplace = 'FinalNearestNeighborInterpolator'

    # read and replace
    with open(transformixParameters, 'r') as fd:
        # sample case-insensitive find-and-replace
        text, counter = re.subn(textToSearch, textToReplace, fd.read(), re.I)

    # check if there is at least a  match
    if counter > 0:
        # edit the file
        with open(transformixParameters, 'w') as fd:
            fd.write(text)


    lab_regs=list(map(partial(transform_label,out_folder=out_folder, transformix_path=transformix_path,transformixParameters=transformixParameters),labels_b_list))


    return (modality,result,lab_regs) #        
 
