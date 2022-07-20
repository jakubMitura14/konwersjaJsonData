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

files_df= mainFuncs.get_df_file_info(dataDir)
files_df
## now performing outer join
#from annotations
#StudyInstanceUID - 199 unique
#SeriesInstanceUID - 743 unique
#from files
#StudyInstanceUID - 3 unique
#SeriesInstanceUID - 44 unique
numpy_StudyInstanceUID=annot['StudyInstanceUID'].to_numpy()
len(np.unique(numpy_StudyInstanceUID))

mergedWithPaths=pd.merge(annot,files_df,on='SOPInstanceUID',how='inner')
mergedWithPaths

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


