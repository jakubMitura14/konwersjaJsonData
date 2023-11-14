from nnunetv2.training.nnUNetTrainer.my_transform import My_PseudoLesion_adder
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
# from .Pl_model import *
import shutil
import h5py
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
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

from transformers import AutoImageProcessor
from ...med_next.create_mednext_v1 import *
from ..swin_organized.SwinUNETR import *

class My_pl_trainer(nnUNetTrainer):

    def on_train_start(self):
        """
        we will additionally invoke here the initialization of pytorch lightning module
        """
        self.log_every_n=5
        self.num_batch_to_eval=10
        train_eval_folder ='/workspaces/konwersjaJsonData/explore/validation_to_look_into/train'
        val_eval_folder ='/workspaces/konwersjaJsonData/explore/validation_to_look_into/val'
        ligtning_logs_folder='/workspaces/konwersjaJsonData/explore'
        h5_folder='/workspaces/konwersjaJsonData/explore/hdf5_loc'
        hf5_path=f"{h5_folder}/mytestfile.hdf5"
        for_explore='/workspaces/konwersjaJsonData/explore/for_explore'
        self.hf5_path=hf5_path
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
        
        nnUNetTrainer.on_train_start(self)


        self.is_plain_conv=True

        self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                self.dataset_json)

        if(self.is_plain_conv):
            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                            self.configuration_manager,
                                                            self.num_input_channels,
                                                            enable_deep_supervision=True).to(self.device)
        # compile network for free speedup




        # self.network=create_mednextv1_large(num_input_channels=self.num_input_channels
        #                                     ,num_classes=self.label_manager.num_segmentation_heads
        #                                     ,kernel_size= 3
        #                                     ,ds= True)
        
        # self.network=SwinUNETR(in_channels=self.num_input_channels
        #             ,out_channels=self.label_manager.num_segmentation_heads
        #             ,use_v2=True#
        #             ,img_size=(96, 96, 96))

        if self._do_i_compile():
            self.print_to_log_file('Compiling network...')
            self.network = torch.compile(self.network)

        # self.save_hyperparameters()
        # self.pl_model= Pl_Model.load_from_checkpoint(self.output_folder)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        # if ddp, wrap in DDP wrapper
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])



        self.pl_model= Pl_Model(network=self.network
                                ,dataloader_train=self.dataloader_train
                                ,dataloader_val=self.dataloader_val
                                ,loss=self.loss
                                ,learning_rate=self.initial_lr
                                ,weight_decay=self.weight_decay
                                ,label_manager=self.label_manager
                                ,log_every_n=self.log_every_n
                                ,num_batch_to_eval=self.num_batch_to_eval
                                ,train_eval_folder=train_eval_folder 
                                ,val_eval_folder=val_eval_folder
                                ,hf5_path=self.hf5_path
                                ,for_explore=for_explore
                                ,is_plain_conv=self.is_plain_conv
                                ,batch_size=self.batch_size)

        comet_logger = CometLogger(
            api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
            #workspace="OPI", # Optional
            project_name=os.getenv('my_proj_name'), # Optional
            #experiment_name="baseline" # Optional
        )


        toMonitor="is_correct_val"
        checkpoint_callback = ModelCheckpoint(dirpath= join(self.output_folder),mode='max', save_top_k=1, monitor=toMonitor)
        # stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=trial.suggest_float("swa_lrs", 1e-6, 1e-4))
        stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=1e-3)
        # optuna_prune=PyTorchLightningPruningCallback(trial, monitor=toMonitor)     
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(
            monitor=toMonitor,
            patience=20,
            mode="max",
            #divergence_threshold=(-0.1)
        )
        # amp_plug=pl.pytorch.plugins.precision.MixedPrecisionPlugin()
        self.trainer = pl.Trainer(
            #accelerator="cpu", #TODO(remove)
            max_epochs=1000,
            #gpus=1,
            #precision=experiment.get_parameter("precision"), 
            callbacks=[ early_stopping,checkpoint_callback], #stochasticAveraging,optuna_prune,checkpoint_callback
            logger=comet_logger,
            accelerator='auto',
            devices='auto',       
            default_root_dir= self.default_root_dir,
            # auto_scale_batch_size="binsearch",
            check_val_every_n_epoch=self.log_every_n,
            accumulate_grad_batches= 10,
            gradient_clip_val = 3.0 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
            log_every_n_steps=self.log_every_n
            # ,reload_dataloaders_every_n_epochs=1
            #strategy='dp'
        )




    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler
    
    def _build_loss(self):
        loss= Picai_FL_and_CE_loss()

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss    

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if(self.is_classic_nnunet):
            if self.is_ddp:
                self.network.module.decoder.deep_supervision = enabled
            else:
                self.network.decoder.deep_supervision = enabled
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(RicianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(My_PseudoLesion_adder())
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        # tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))




        tr_transforms.append(RenameTransform('seg', 'target', True))



        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms


    def run_training(self):
        self.on_train_start()

        # tuner = Tuner(self.trainer)
        # to set to your own hparams.my_value
        # tuner.lr_find(self.pl_model, attr_name="learning_rate")
        self.trainer.fit(self.pl_model)
        
        self.on_train_end()
        # shutil.rmtree(self.default_root_dir)
        # self.f.close()

