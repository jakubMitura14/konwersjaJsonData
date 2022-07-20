import SimpleITK as sitk
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
import labelDict
import random





mdai_client = mdai.Client(domain='public.md.ai', access_token="1d48dd3c05ce6d59759915d9328fe769")
p = mdai_client.project('gaq3y0Rl', path='/workspaces/konwersjaJsonData/data')
p.set_labels_dict(labels_dict)
datasetId='D_8og9ON'
dataset = p.get_dataset_by_id(datasetId)
dataset.prepare()

image_filenames = dataset.get_image_ids()
fn = image_filenames[0]
image = pydicom.dcmread(fn)
mask = mdai.visualize.load_mask(fn, dataset)
img = image.pixel_array


image_id = random.choice(dataset.get_image_ids())
image, class_ids, bboxes, masks = mdai.visualize.get_image_ground_truth(image_id, dataset)



image_filenames = dataset.get_image_ids()
fn = image_filenames[0]
correctPath='/workspaces/konwersjaJsonData/data/mdai_public_project_gaq3y0Rl_images_2022-07-19-115923_part1of2/1.3.12.2.1107.5.8.15.100960.30000021071711593919300001006/1.3.12.2.1107.5.8.15.100960.30000021071711593919300001035/1.3.12.2.1107.5.8.15.100960.30000021071711593919300001051.dcm'
image = pydicom.dcmread(correctPath)
mask = mdai.visualize.load_mask(fn, dataset)




image_id = random.choice(dataset.get_image_ids())
pixel_array = mdai.visualize.load_dicom_image(image_id, to_RGB=False, rescale=True)



p.show_label_groups()
labels_dict=labelDict.labels_dict
p.set_labels_dict(labels_dict)

p.show_datasets()
dataset = p.get_dataset_by_id(datasetId)
dataset.prepare()

dataset.show_classes()


#we need to keep part private for validation purposes ?
train_dataset, valid_dataset = mdai.common_utils.train_test_split(dataset)

image_id = random.choice(dataset.get_image_ids())

pixel_array = mdai.visualize.load_dicom_image(image_id, to_RGB=False, rescale=True)


mask = mdai.visualize.load_mask(image_id, dataset)
image_plus_mask = mdai.visualize.apply_mask(image, mask, color, alpha=0.3)



original_image,gt_class_id,gt_box,gt_mask=mdai.visualize.get_image_ground_truth(image_id,dataset)



mask = mdai.visualize.load_mask(image_id, dataset)
image_plus_mask = mdai.visualize.apply_mask(image, mask, color, alpha=0.3)

image, class_ids, bboxes, masks = mdai.visualize.get_image_ground_truth(image_id, dataset)


token iryna 1d48dd3c05ce6d59759915d9328fe769

OPI PIB INFOSTRATEG - AI4AR
nazwa
D_8og9ON
dataset1


D_gQm1nQ
dataset2

project_id - gaq3y0Rl