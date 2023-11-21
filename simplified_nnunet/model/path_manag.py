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
import shutil
import h5py
import lightning.pytorch as pl
# from lightning.pytorch.tuner import Tuner
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from lightning.pytorch.callbacks import LearningRateFinder

def manage_paths():
       train_eval_folder ='/workspaces/konwersjaJsonData/explore/validation_to_look_into/train'
        val_eval_folder ='/workspaces/konwersjaJsonData/explore/validation_to_look_into/val'
        ligtning_logs_folder='/workspaces/konwersjaJsonData/explore'
        h5_folder='/workspaces/konwersjaJsonData/explore/hdf5_loc'
        hf5_path=f"{h5_folder}/mytestfile.hdf5"
        for_explore='/workspaces/konwersjaJsonData/explore/for_explore'
        self.hf5_path=hf5_path
        attn_masks_h5f_path="/workspaces/konwersjaJsonData/sparse_dat/sparse_masks.hdf5"

        os.makedirs(h5_folder,exist_ok=True)


        os.makedirs(train_eval_folder,exist_ok=True)
        os.makedirs(val_eval_folder,exist_ok=True)
        os.makedirs(ligtning_logs_folder,exist_ok=True)
        os.makedirs(for_explore,exist_ok=True)
        
        
        shutil.rmtree(h5_folder)        
        shutil.rmtree(train_eval_folder)
        shutil.rmtree(val_eval_folder)
        shutil.rmtree(ligtning_logs_folder)
        # shutil.rmtree(for_explore)
        
        
        os.makedirs(train_eval_folder,exist_ok=True)        
        os.makedirs(ligtning_logs_folder,exist_ok=True)
        os.makedirs(val_eval_folder,exist_ok=True)
        os.makedirs(h5_folder,exist_ok=True)
        os.makedirs(for_explore,exist_ok=True)

        self.default_root_dir=ligtning_logs_folder
