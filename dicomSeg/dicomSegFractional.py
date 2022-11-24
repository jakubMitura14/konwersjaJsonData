# from pathlib import Path

# import highdicom as hd
# import numpy as np
# from pydicom.sr.codedict import codes
# from pydicom.filereader import dcmread
# import pandas as pd
# import SimpleITK as sitk
# from pathlib import Path

# import highdicom as hd
# import numpy as np
# from pydicom.sr.codedict import codes
# from pydicom.filereader import dcmread
# from os.path import basename, dirname, exists, isdir, join, split
# import sys
# import time
# import os
# import numpy as np
# import pathlib
# import pydicom

# #directory of the csv storing most relevant data per each series
# resCSVDir='/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
# #directory for json with dicom seg definitions of prostate regions
# dseg_anatomy_json='/workspaces/konwersjaJsonData/dicomSeg/anatomy_seg.json'
# #directory for json with dicom seg definitions of lesion regions
# dseg_lesion_json='/workspaces/konwersjaJsonData/dicomSeg/lesion_seg.json'
# #names of columns used to create dicom seg
# innerLabel_json_names=['anterior_fibromuscular_stroma','central_zone','external_iliac','internal_iliac','lesion1','lesion2','lesion3','lesion4','obturator','peripheral_zone','prostate',
# 'seminal_vesiclesL','seminal_vesiclesR','transition_zone','urethra']
# #names without space that is consistent with the name of json files and their internal names with the same order as innerLabel_json_names
# col_names_to_analyze = ['anterior fibromuscular stroma', 'central zone', 'external iliac', 'internal iliac', 'obturator', 'peripheral zone', 'prostate', 'seminal vesicles L', 'seminal vesicles R', 'transition zone', 'urethra']
# lesionColumns=['lesion 1', 'lesion 2', 'lesion 3', 'lesion 4']
# #folder with jsopns for specifications with dicom seg
# jsonFolder='/workspaces/konwersjaJsonData/dicomSeg'

# prostateLab='prostate'
# out_files_frame=pd.read_csv(resCSVDir) 
# frame_of_intr=out_files_frame.loc[out_files_frame[prostateLab]!=" "]
# roww = list(frame_of_intr.iterrows())[0]



# def get_SOPInstanceUID(ds):
#     return ds.file_meta[(0x0002, 0x0003)].repval
# def get_SeriesInstanceUID(ds):
#     return ds[(0x0020, 0x000e)].repval

# roww=roww[1]
# folderWithDicoms=roww['series_MRI_path'].replace('volume.mha','origVol')
# origDicoms = list(pathlib.Path(folderWithDicoms).glob('*.dcm'))
# basePath="/workspaces/konwersjaJsonData/exploration"
# colName="prostate"
# colName=colName.replace(" ","_")
# colFolder= join(basePath,colName)
# os.makedirs(colFolder ,exist_ok = True)
# series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folderWithDicoms, roww[1]['series_id'])


# for pathTupl in enumerate(series_file_names):
#     index,path=pathTupl
# # path = origDicoms[0]
#     ds = pydicom.dcmread(path)
#     currSop = get_SOPInstanceUID(ds)
#     currSeriesId = get_SeriesInstanceUID(ds)
#     saveMonoDicomSeg(roww,colFolder,colName,currSop,currSeriesId,path,index)



# def get_dataset(roww, colName,num,colorCode,algorithm_identification,index):
#     image = sitk.ReadImage(roww[colName] )
#     mask=sitk.GetArrayFromImage(image)
#     print(f" mask shape {mask.shape}")
#     mask = mask[index,:,:]
#     # mask= mask-np.min(mask.flatten())
#     # mask= mask/np.max(mask.flatten())
#     # Describe the segment
#     description_segment_1 = hd.seg.SegmentDescription(
#         segment_number=num,
#         segment_label=colName,
#         segmented_property_category=codes.cid7150.Tissue,
#         segmented_property_type=colorCode,
#         algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
#         algorithm_identification=algorithm_identification,
#         tracking_uid=hd.UID(),
#         tracking_id=colName
#     )
#     return mask,description_segment_1


# def saveMonoDicomSeg(roww,colFolder,colname,currSop,currSeriesId, singleDcmFile,index):
#     roww= roww[1]
#     outputPath=join(colFolder,f"{currSop}.dcm") 
#     # Read CT Image data sets from PS3.10 files on disk
#     image_datasets = [dcmread(str(singleDcmFile))]

#     # Describe the algorithm that created the segmentation
#     algorithm_identification = hd.AlgorithmIdentificationSequence(
#         name='algorithm',
#         version='v1.0',
#         family=codes.cid7162.ArtificialIntelligence)
#     mask1,desc1=get_dataset(roww, colname,1,codes.cid7166.ConnectiveTissue,algorithm_identification,index)

#     # Create the Segmentation instance
#     seg_dataset = hd.seg.Segmentation(
#         source_images=image_datasets,
#         pixel_array=mask1.astype('uint8'),
#         segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
#         # segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
#         segment_descriptions=[desc1],
#         series_instance_uid=hd.UID(),#currSeriesId,
#         series_number=1,
#         # content_label=colName,
#         content_description=colName,
#         sop_instance_uid=currSop,
#         instance_number=1,
#         manufacturer='Manufacturer',
#         manufacturer_model_name='Model',
#         software_versions='v1',
#         omit_empty_frames=False,
#         device_serial_number='Device XYZ')

#     seg_dataset.save_as(join(outputPath))


# colname= "prostate"
# outputPath="/workspaces/konwersjaJsonData/exploration"
# saveMonoDicomSeg(roww,outputPath,colname)
# saveMonoDicomSeg(roww,outputPath,"anterior fibromuscular stroma")
# saveMonoDicomSeg(roww,outputPath,"central zone")
# saveMonoDicomSeg(roww,outputPath,"lesion 1")






# # def saveMonoDicomSeg(roww,outputPath,colname):
# #     roww= roww[1]
# #     folderWithDicoms=roww['series_MRI_path'].replace('volume.mha','origVol')
# #     print(f"folder with mri {folderWithDicoms}")
# #     series_dir = Path(folderWithDicoms)
# #     image_files = series_dir.glob('*.dcm')
# #     # Read CT Image data sets from PS3.10 files on disk
# #     image_datasets = [dcmread(str(f)) for f in image_files]

# #     # Describe the algorithm that created the segmentation
# #     algorithm_identification = hd.AlgorithmIdentificationSequence(
# #         name='test',
# #         version='v1.0',
# #         family=codes.cid7162.ArtificialIntelligence)
# #     mask1,desc1=get_dataset(roww, colname,1,codes.cid7166.ConnectiveTissue,algorithm_identification)

# #     # Create the Segmentation instance
# #     seg_dataset = hd.seg.Segmentation(
# #         source_images=image_datasets,
# #         pixel_array=mask1.astype('uint8'),
# #         segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
# #         # segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
# #         segment_descriptions=[desc1],
# #         series_instance_uid=hd.UID(),
# #         series_number=1,
# #         sop_instance_uid=hd.UID(),
# #         instance_number=1,
# #         manufacturer='Manufacturer',
# #         manufacturer_model_name='Model',
# #         software_versions='v1',
# #         omit_empty_frames=False,
# #         device_serial_number='Device XYZ')

# #     colnameNew=colname.replace(" ","_")
# #     seg_dataset.save_as(join(outputPath,f"{colnameNew}.dcm"))

# # colname= "prostate"
# # outputPath="/workspaces/konwersjaJsonData/exploration"
# # saveMonoDicomSeg(roww,outputPath,colname)
# # saveMonoDicomSeg(roww,outputPath,"anterior fibromuscular stroma")
# # saveMonoDicomSeg(roww,outputPath,"central zone")
# # saveMonoDicomSeg(roww,outputPath,"lesion 1")

# # anterior fibromuscular stroma

# # masks = np.stack([mask1,mask2,mask3],axis=3 ).astype('uint8')

# # # Create the Segmentation instance
# # seg_dataset = hd.seg.Segmentation(
# #     source_images=image_datasets,
# #     pixel_array=mask1.astype('uint8'),
# #     segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
# #     # segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
# #     segment_descriptions=[desc1 ],
# #     series_instance_uid=hd.UID(),
# #     series_number=1,
# #     sop_instance_uid=hd.UID(),
# #     instance_number=1,
# #     manufacturer='Manufacturer',
# #     manufacturer_model_name='Model',
# #     software_versions='v1',
# #     device_serial_number='Device XYZ')



# # mask1,desc1=get_dataset(roww, "prostate",1,codes.cid7166.ConnectiveTissue)
# # mask2,desc2=get_dataset(roww, "central zone",2,codes.cid7166.ConnectiveTissue)
# # mask3,desc3=get_dataset(roww, "peripheral zone",3,codes.cid7166.Artery)
# # mask4,desc4=get_dataset(roww, "anterior fibromuscular stroma",4,codes.cid7166.Organ)
# # # mask4,desc4=get_dataset(roww, "series_MRI_path",4,codes.cid7166.Organ)


# # print(seg_dataset)

# # seg_dataset.save_as("/workspaces/konwersjaJsonData/exploration/seg.dcm")



# # # def writeSlices(series_tag_values, new_img, out_dir, i,writer):
# # #     """
# # #     copied from https://simpleitk.readthedocs.io/en/master/link_DicomSeriesFromArray_docs.html
# # #     """
# # #     image_slice = new_img[:, :, i]

# # #     # Tags shared by the series.
# # #     list(
# # #         map(
# # #             lambda tag_value: image_slice.SetMetaData(
# # #                 tag_value[0], tag_value[1]
# # #             ),
# # #             series_tag_values,
# # #         )
# # #     )

# # #     # Slice specific tags.
# # #     #   Instance Creation Date
# # #     image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
# # #     #   Instance Creation Time
# # #     image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

# # #     # Setting the type to CT so that the slice location is preserved and
# # #     # the thickness is carried over.
# # #     image_slice.SetMetaData("0008|0060", "CT")

# # #     # (0020, 0032) image position patient determines the 3D spacing between
# # #     # slices.
# # #     #   Image Position (Patient)
# # #     image_slice.SetMetaData(
# # #         "0020|0032",
# # #         "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))),
# # #     )
# # #     #   Instance Number
# # #     image_slice.SetMetaData("0020,0013", str(i))

# # #     # Write to the output directory and add the extension dcm, to force
# # #     # writing in DICOM format.
# # #     writer.SetFileName(os.path.join(out_dir, str(i) + ".dcm"))
# # #     writer.Execute(image_slice)

# # # def writeAllSlices():
# # #     image=
# # #     pixel_dtype=
# # #     writer = sitk.ImageFileWriter()
# # #     # Use the study/series/frame of reference information given in the meta-data
# # #     # dictionary and not the automatically generated information from the file IO
# # #     writer.KeepOriginalImageUIDOn()
# # #     direction = image.GetDirection()
# # #     modification_time = time.strftime("%H%M%S")
# # #     modification_date = time.strftime("%Y%m%d")
# # #     series_tag_values = [
# # #         ("0008|0031", modification_time),  # Series Time
# # #         ("0008|0021", modification_date),  # Series Date
# # #         ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
# # #         (
# # #             "0020|000e",
# # #             "1.2.826.0.1.3680043.2.1125."
# # #             + modification_date
# # #             + ".1"
# # #             + modification_time,
# # #         ),  # Series Instance UID
# # #         (
# # #             "0020|0037",
# # #             "\\".join(
# # #                 map(
# # #                     str,
# # #                     (
# # #                         direction[0],
# # #                         direction[3],
# # #                         direction[6],
# # #                         direction[1],
# # #                         direction[4],
# # #                         direction[7],
# # #                     ),
# # #                 )
# # #             ),
# # #         ),  # Image Orientation
# # #         # (Patient)
# # #         ("0008|103e", "Created-SimpleITK"),  # Series Description
# # #     ]
# # #     if pixel_dtype == np.float64:
# # #         # If we want to write floating point values, we need to use the rescale
# # #         # slope, "0028|1053", to select the number of digits we want to keep. We
# # #         # also need to specify additional pixel storage and representation
# # #         # information.
# # #         rescale_slope = 0.001  # keep three digits after the decimal point
# # #         series_tag_values = series_tag_values + [
# # #             ("0028|1053", str(rescale_slope)),  # rescale slope
# # #             ("0028|1052", "0"),  # rescale intercept
# # #             ("0028|0100", "16"),  # bits allocated
# # #             ("0028|0101", "16"),  # bits stored
# # #             ("0028|0102", "15"),  # high bit
# # #             ("0028|0103", "1"),
# # #         ]  # pixel representation


