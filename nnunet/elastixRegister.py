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
import itk


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
    labels_b_list= list(labels_b_list)

    cmd=f"{elacticPath} -f {path_a} -m {path} -out {outPath} -p {reg_prop} -threads 1"
    p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
    p.wait()
    #we will repeat operation multiple max 9 times if the result would not be written
    if((not pathOs.exists(result)) and reIndex<5):
       
        reg_prop=reg_prop.replace("parameters","parametersB")

        cmd=f"{elacticPath} -f {path_a} -m {path} -out {outPath} -p {reg_prop} -threads 1"

        # p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
        p = Popen(cmd, shell=True)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE

        p.wait()

        reIndexNew=reIndex+1
        # if(reIndex==1): #in case it do not work we will try diffrent parametrization
        #     reg_prop=reg_prop.replace("parameters","parametersB")              
        # #recursively invoke function multiple times in order to maximize the probability of success    
        # reg_a_to_b(out_folder,patId,path_a,path_b,labels_b_list,reg_prop ,elacticPath,transformix_path,modality,reIndexNew)
    if(not pathOs.exists(result)):
        print(f"registration unsuccessfull {patId}")
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


    lab_regs=list(map(partial(transform_label,out_folder=out_folder, transformix_path=transformix_path,transformixParameters=transformixParameters),np.array(labels_b_list).flatten()))


    return (modality,result,lab_regs) #        
 

def apply_itk_transformix(to_transform_path,moving_image,result_transform_parameters,out_folder):
    """
    applying transformix 
    """
    moving_image_transformix = itk.imread(to_transform_path, itk.F)
    result_image_transformix = itk.transformix_filter(
        moving_image_transformix,
        result_transform_parameters
        , log_to_console=True)
    path_save=f"{out_folder}/{os.path.basename(to_transform_path)}"
    itk.imwrite(result_image_transformix,path_save)
    return path_save


def reg_a_to_b_itk(out_folder,patId,path_a,path_b,labels_b_list,reg_prop ,elacticPath,transformix_path,modality,reIndex=0):
    """
    reg_a_to_b version using itk elastix
    """
    fixed_image =  sitk.ReadImage(path_a, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(path_b, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)
    optimized_transform = sitk.Euler3DTransform()    
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=False)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        
    sitk.WriteImage(moving_resampled, path_a)

    fixed_image = itk.imread(path_a, itk.F)
    moving_image = itk.imread(path_b, itk.F)

    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()
    # parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('affine')
    

    parameter_object.AddParameterMap(parameter_map_rigid)

    transformed_a, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,log_to_console=True)

    transformed_a_path=itk.imwrite(transformed_a,f"{out_folder}/result_image_a.mha")

    labels_b_list= list(labels_b_list)
    np.array(labels_b_list).flatten()

    lab_regs=list(map(partial(apply_itk_transformix,moving_image=moving_image,result_transform_parameters=result_transform_parameters,out_folder=out_folder ),np.array(labels_b_list).flatten()))


    return (modality,transformed_a_path,lab_regs) #     