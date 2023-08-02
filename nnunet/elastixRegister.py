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
import itertools


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
       
        # reg_prop=reg_prop.replace("parameters","parametersB")

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



def reg_a_to_b_by_metadata_single(fixed_image_path,moving_image_path,out_folder):
    """
    adapted from last section of
    https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/21_Transforms_and_Resampling.ipynb 
    """
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)

    # identity = sitk.Transform(3, sitk.sitkIdentity)


    images = [fixed_image, moving_image]
    transforms = [sitk.Transform(3, sitk.sitkIdentity),sitk.Transform(3, sitk.sitkIdentity)]
    dim = images[0].GetDimension()

    boundary_points = []
    for image, transform in zip(images, transforms):
        for boundary_index in list(
            itertools.product(*zip([0] * dim, [sz - 1 for sz in image.GetSize()]))
        ):  # Points from the moving image(s) are mapped to the fixed coordinate system using the inverse
            # of the registration_result.
            boundary_points.append(
                # transform.GetInverse().TransformPoint(
                #     image.TransformIndexToPhysicalPoint(boundary_index)
                # )
                image.TransformIndexToPhysicalPoint(boundary_index)
            )
    max_coords = np.max(boundary_points, axis=0)
    min_coords = np.min(boundary_points, axis=0)

    new_origin = min_coords
    # Arbitrarily use the spacing of the first image and its pixel type,
    # change these to suite your needs.
    new_spacing = images[0].GetSpacing()
    new_pixel_type = images[0].GetPixelID()
    new_size = (((max_coords - min_coords) / new_spacing).round().astype(int)).tolist()
    new_direction = np.identity(dim).ravel()

    # Resample all images onto the common grid.
    resampled_images = []
    for image, transform in zip(images, transforms):
        resampled_images.append(
            sitk.Resample(
                image,
                new_size,
                transform,
                sitk.sitkLinear,
                new_origin,
                new_spacing,
                new_direction,
                0.0,
                new_pixel_type,
            )
        )
    os.makedirs(out_folder ,exist_ok = True)   
    writer = sitk.ImageFileWriter()

    arr=sitk.GetArrayFromImage(resampled_images[1])
    print(f"leen {len(resampled_images)} \n prim sum {np.sum(sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path)).flatten())} \n suuum {np.sum(arr.flatten())} \n max_coords {max_coords} \n min_coords {min_coords} \n")

    new_path= join(out_folder,moving_image_path.split('/')[-1])
    writer.SetFileName(new_path)
    writer.Execute(resampled_images[1])

    return new_path

def reg_a_to_b_by_metadata_single_b(fixed_image_path,moving_image_path,out_folder):
    moving_image_path=moving_image_path[0]
    fixed_image=sitk.ReadImage(fixed_image_path)
    moving_image=sitk.ReadImage(moving_image_path)

    # fixed_image=sitk.Cast(fixed_image, sitk.sitkUInt8)
    # moving_image=sitk.Cast(moving_image, sitk.sitkInt)
    
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0)
    
    # print(f" prim sum {np.sum(sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path)).flatten())} \n suuum {np.sum(sitk.GetArrayFromImage(resampled).flatten())} ")
  
    writer = sitk.ImageFileWriter()
    new_path= join(out_folder,moving_image_path.split('/')[-1])
    writer.SetFileName(new_path)
    writer.Execute(resampled)

    return new_path

# def reg_a_to_b_by_metadata(out_folder,patId,path_a,path_b,labels_b_list,reg_prop ,elacticPath,transformix_path,modality,reIndex=0):
#     pass
    
    
# # curr_adc= '/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont/Data/079/79_adc.mha'
# curr_adc= '/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_dicom/Data/079/79_adc'
# curr_afs= '/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont/Anatomical_Labels/079/79_afs_t2w.nii.gz'

# # curr_adc= '/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont/Data/066/66_adc.mha'
# # curr_afs= '/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont/Anatomical_Labels/066/66_afs_t2w.nii.gz'

# out_folder='/workspaces/konwersjaJsonData/explore/temp'
# reg_a_to_b_by_metadata_single_b(curr_adc,curr_afs,out_folder)

