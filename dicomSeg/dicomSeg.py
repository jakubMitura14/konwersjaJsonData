#multilabel segmentation https://razorx89.github.io/pydicom-seg/guides/seg_types.html
#newer package nifti to dicom seg https://github.com/deepc-health/nekton
import pydicom
import pydicom_seg
import SimpleITK as sitk
import pandas as pd
from os.path import basename, dirname, exists, isdir, join, split
import highdicom as hd
import numpy as np
from pydicom.sr.codedict import codes
from pydicom.filereader import dcmread
from mainFuncs import getLabelsAbbrev

def save_dicom_seg_label(roww,jsonFolder,colName,outPath=' ',dicomSPath=' ',seriesId=' ',segmentation=' ',curr_no_space_name=' ' ):
    """
    saves single label from single series into dicom seg format
    row - row in dataframes of intrest
    index - points to which label from supplied innerLabel_json_names and col_names_to_analyze entry to refer
    """
    #name to retrieve json
    if(curr_no_space_name==' '):
        curr_no_space_name=colName.replace('_noSeg','')
    else:
        curr_no_space_name=getLabelsAbbrev(curr_no_space_name)
    colNameSeg=f"{curr_no_space_name}_Seg"
    #getting path to json in order to get template specification and instantiate dicom seg writer
    currentJson_path=join(jsonFolder,curr_no_space_name+".json") 
    template_curr= pydicom_seg.template.from_dcmqi_metainfo(currentJson_path)
    writer = pydicom_seg.MultiClassWriter(
        template=template_curr,
        inplane_cropping=False,  # Crop image slices to the minimum bounding box on
                                # x and y axes
        skip_empty_slices=False,  # Don't encode slices with only zeros
        skip_missing_segment=False,  # If a segment definition is missing in the
                                    # template, then raise an error instead of
                                    # skipping it.
    )
    #retrieving segmentation label
    if(segmentation==' '):
        segmentation=sitk.ReadImage(roww[colName])
    #getting folder with original dicoms of the MRI
    #read oridginal dicoms
    reader = sitk.ImageSeriesReader()

    if(dicomSPath==' '):
        dicomSPath=roww['dicom_MRI_path']
    if(seriesId==' '):
        seriesId=roww['series_id']  

    dcm_files = reader.GetGDCMSeriesFileNames(dicomSPath, seriesId)
    source_images = [
        pydicom.dcmread(x, stop_before_pixels=True)
        for x in dcm_files
    ]
    #save segmentation in dicom seg format to path
    dcm = writer.write(sitk.Cast(segmentation, sitk.sitkUInt8), source_images)
    if(outPath==' '):
        outPath=roww[colNameSeg]
    
    dcm.save_as(outPath)
    # dcm.save_as(join(roww[colNameSeg], curr_no_space_name+".dcm"))
