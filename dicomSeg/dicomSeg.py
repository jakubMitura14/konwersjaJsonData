#multilabel segmentation https://razorx89.github.io/pydicom-seg/guides/seg_types.html
#newer package nifti to dicom seg https://github.com/deepc-health/nekton
import pydicom
import pydicom_seg
import SimpleITK as sitk
import pandas as pd
from os.path import basename, dirname, exists, isdir, join, split

#directory of the csv storing most relevant data per each series
resCSVDir='/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
#directory for json with dicom seg definitions of prostate regions
dseg_anatomy_json='/workspaces/konwersjaJsonData/dicomSeg/anatomy_seg.json'
#directory for json with dicom seg definitions of lesion regions
dseg_lesion_json='/workspaces/konwersjaJsonData/dicomSeg/lesion_seg.json'
#names of columns used to create dicom seg
innerLabel_json_names=['anterior_fibromuscular_stroma','central_zone','external_iliac','internal_iliac','lesion1','lesion2','lesion3','lesion4','obturator','peripheral_zone','prostate',
'seminal_vesiclesL','seminal_vesiclesR','transition_zone','urethra']
#names without space that is consistent with the name of json files and their internal names with the same order as innerLabel_json_names
col_names_to_analyze = ['anterior fibromuscular stroma', 'central zone', 'external iliac', 'internal iliac', 'lesion 1', 'lesion 2', 'lesion 3',
       'lesion 4',  'obturator', 'peripheral zone', 'prostate', 'seminal vesicles L', 'seminal vesicles R', 'transition zone', 'urethra']
#folder with jsopns for specifications with dicom seg
jsonFolder='/workspaces/konwersjaJsonData/dicomSeg'

prostateLab='prostate'

# out_files_frame=pd.read_csv(resCSVDir) 



# index=0

def save_dicom_seg_label(row,index,innerLabel_json_names,col_names_to_analyze, path_to_save):
    """
    saves single label from single series into dicom seg format
    row - row in dataframes of intrest
    index - points to which label from supplied innerLabel_json_names and col_names_to_analyze entry to refer
    """
    roww=row[1]
    #name to retrieve json
    curr_no_space_name=innerLabel_json_names[index]
    #name as column name in pandas dataframe
    curr_with_space_name=col_names_to_analyze[index]
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
    segmentation=sitk.ReadImage(roww[curr_with_space_name])
    #getting folder with original dicoms of the MRI
    folderWithDicoms=roww['series_MRI_path'].replace('volume.mha','origVol')
    #read oridginal dicoms
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(folderWithDicoms, roww['series_id'])
    source_images = [
        pydicom.dcmread(x, stop_before_pixels=True)
        for x in dcm_files
    ]
    #save segmentation in dicom seg format to path
    dcm = writer.write(sitk.Cast(segmentation, sitk.sitkUInt8), source_images)
    dcm.save_as(join(path_to_save, curr_no_space_name+".dcm"))


#     roww[curr_with_space_name]


#     frame_of_intr=out_files_frame.loc[out_files_frame[prostateLab]!=" "]
#     row = list(frame_of_intr.iterrows())[0]
# roww['series_MRI_path']



# #image_mri = sitk.ReadImage(roww['series_MRI_path'])

# mriImage=sitk.ReadImage(pathA)
# out_files_frame.columns

# writer_anatomy

# # image_orig = sitk.ReadImage(pathA)
# # image_data_orig = sitk.GetArrayFromImage(image)




# # template_lesion = pydicom_seg.template.from_dcmqi_metainfo(dseg_lesion_json)

# # writer_anatomy = pydicom_seg.MultiClassWriter(
# #     template=template_anatomy,
# #     inplane_cropping=False,  # Crop image slices to the minimum bounding box on
# #                             # x and y axes
# #     skip_empty_slices=False,  # Don't encode slices with only zeros
# #     skip_missing_segment=False,  # If a segment definition is missing in the
# #                                  # template, then raise an error instead of
# #                                  # skipping it.
# # )


# #     def segments(self) -> List[Segment]:
# #         """
# #         Segments property.
# #         Returns
# #         -------
# #         segments : List[Segment]
# #             List of all the segments.
# #         """
# #         reader =pydicom_seg.SegmentReader()

# #         segments = []
# #         for segment_number, dicom_header in result.segment_infos.items():
# #             if dicom_header.__contains__("SegmentLabel"):
# #                 organ_name = dicom_header.SegmentLabel
# #             else:
# #                 organ_name = dicom_header.SegmentDescription

# #             simple_itk_label_map = result.segment_image(segment_number)

# #             segments.append(Segment(name=organ_name, simple_itk_label_map=simple_itk_label_map))

# #         return segments