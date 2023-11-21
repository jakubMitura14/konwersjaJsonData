from nnunetv2.training.nnUNetTrainer.my_transform import My_PseudoLesion_adder,My_priming_setter
from nnunetv2.training.nnUNetTrainer.custom_loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
# from focal_loss.focal_loss import FocalLoss
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import RicianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from ..Main_pl_model import *
import shutil
import h5py
import lightning.pytorch as pl
# from lightning.pytorch.tuner import Tuner
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP



import transformers
from mpi4py import MPI
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .med_next.create_mednext_v1 import *
from .swin_unetr.swin_organized.SwinUNETR import *

def select_model(img_size
                 ,is_med_next
                 ,is_swin
                 ,is_swin_monai
                 ,is_classic_nnunet
                 ,dataset_json
                 ,configuration_manager
                 ,plans_manager
                 ,label_manager
                 ,num_input_channels
                 ,batch_size
                 ,deep_supervision):
    """
    selecting the model - generally we are choosing from 3 models:
    swin unetr, med next and classic nnunet
    """

    if(is_classic_nnunet):
        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                    num_input_channels, deep_supervision=True)
        if(deep_supervision):
            network.decoder.deep_supervision = True 

    if(is_med_next):
        network=create_mednextv1_large(num_input_channels=num_input_channels
                                            ,num_classes=label_manager.num_segmentation_heads
                                            ,kernel_size= 7
                                            ,ds= True)
    if(is_swin_monai):
        img_size=(64,img_size[1],img_size[2])
        # attn_masks_h5f=h5py.File(attn_masks_h5f_path,'w') 
        network=SwinUNETR(in_channels=num_input_channels
        # ,num_heads=  (1, 3, 6, 12)
            ,num_heads=  (6, 12, 24, 48)
        ,out_channels=label_manager.num_segmentation_heads
        ,use_v2=True#
        ,img_size=img_size
        ,patch_size=(2,2,2)
        ,batch_size=batch_size
        ,attn_masks_h5f=[]
        ,is_swin=False
        ,is_local_iso=False
        ,is_local_non_iso=False
        ,distances=(7,7,7)
        ,spacing=(3.299999952316284,0.78125, 0.78125)
        ,feature_size=48
        ,depths=(2,2,2,2)
        ,is_lucid=True
        ,window_size=(7,7,7)
        ,use_checkpoint=True
        )


    network = torch.compile(network)

