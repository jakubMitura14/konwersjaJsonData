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
from .Main_pl_model import *
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

from transformers import AutoImageProcessor
from .med_next.create_mednext_v1 import *
from .swin_unetr.swin_organized.SwinUNETR import *
from lightning.pytorch.callbacks import LearningRateFinder

from .swin_unetr.old.Swin_unetr_model_old import SwinUNETR_old

from lightning.pytorch.strategies.ddp import DDPStrategy
from monai.networks.nets.swin_unetr import SwinUNETR as SwinUNETR_monai

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

class Main_trainer_pl(nnUNetTrainer):

    def on_train_start(self):
        """
        we will additionally invoke here the initialization of pytorch lightning module
        """
        self.log_every_n=5
        self.num_batch_to_eval=20
        self.batch_size=1
        self.is_deep_supervision=True
        self.is_classic_nnunet=False
        self.is_swin=False
        self.is_swin_monai=False
        self.is_med_next=True

        self.is_lesion_segm=True
        self.is_anatomy_segm= not self.is_lesion_segm
        self.is_priming_segm= False

        # if(self.is_classic_nnunet or self.is_med_next):
        #     self.is_deep_supervision=True


        self.learning_rate=self.initial_lr 
        if(self.is_swin_monai and self.is_anatomy_segm):
            self.learning_rate=0.02089296130854041
        if(self.is_classic_nnunet and self.is_anatomy_segm):    
            self.learning_rate=0.04365158322401657
        if(self.is_med_next and self.is_anatomy_segm):    #3.77s/it
            self.learning_rate=0.0013182567385564075
                        

        if(self.is_med_next and self.is_lesion_segm):   
            self.learning_rate=0.00831


        self.hparams_dict={"attn_num_mem_kv":os.getenv('attn_num_mem_kv')
                           ,"use_scalenorm":os.getenv('use_scalenorm')
                           ,"sandwich_norm":os.getenv('sandwich_norm')
                           ,"ff_swish":os.getenv('ff_swish')
                           ,"ff_relu_squared":os.getenv('ff_relu_squared')
                           ,"attn_talking_heads":os.getenv('attn_talking_heads')
                           ,"attn_on_attn":os.getenv('attn_on_attn')
                           ,"attn_gate_values":os.getenv('attn_gate_values')
                           ,"sandwich_coef":os.getenv('sandwich_coef')
                           ,"macaron":os.getenv('macaron')
                           ,"residual_attn":os.getenv('residual_attn')
                           ,"gate_residual":os.getenv('gate_residual')
                           ,"shift_tokens":os.getenv('shift_tokens')
                           ,"attn_head_scale":os.getenv('attn_head_scale')
                           ,"ff_post_act_ln":os.getenv('ff_post_act_ln')
                           ,"scale_residual":os.getenv('scale_residual')
                           ,"attn_qk_norm":os.getenv('attn_qk_norm')
                           ,"attn_qk_norm_dim_scale":os.getenv('attn_qk_norm_dim_scale')
                           ,"encoders_depth":os.getenv('encoders_depth')
                           ,"num_memory_tokens":os.getenv('num_memory_tokens')
                           ,"shift_mem_down":os.getenv('shift_mem_down')
                           ,"num_memory_tokens":os.getenv('num_memory_tokens')                            
                             }



        
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
        
        nnUNetTrainer.on_train_start(self)

        if(self.is_lesion_segm):
            self.loss =self._build_loss_lesions()
        elif(self.is_deep_supervision):
            self.loss = self._build_loss()
        elif(self.is_anatomy_segm):
            self.loss=DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)



        self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                self.dataset_json)

        if(self.is_classic_nnunet):
            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                            self.configuration_manager,
                                                            self.num_input_channels,
                                                            enable_deep_supervision=True).to(self.device)
            if self.is_ddp:
                self.network.module.decoder.deep_supervision = True
            else:
                self.network.decoder.deep_supervision = True
        
        img_size=(48, 192, 160)
        if(self.is_lesion_segm):
            img_size=(40, 96, 96)
        

        if(self.is_med_next):
            self.network=create_mednextv1_large(num_input_channels=self.num_input_channels
                                                ,num_classes=self.label_manager.num_segmentation_heads
                                                ,kernel_size= 7
                                                ,ds= True)
        if(self.is_swin):
            attn_masks_h5f=h5py.File(attn_masks_h5f_path,'w') 

            self.network=SwinUNETR(in_channels=self.num_input_channels
            # ,num_heads=  (1, 3, 6, 12)
            # ,num_heads=  (1, 1, 1, 1)
            ,out_channels=self.label_manager.num_segmentation_heads
            ,use_v2=True#
            ,img_size=img_size
            ,patch_size=(1,1,1)
            ,batch_size=self.batch_size
            ,attn_masks_h5f=attn_masks_h5f
            ,is_swin=False
            ,is_local_iso=False
            ,is_local_non_iso=True
            # ,distances=(8,8,16)
            ,distances=(7,7,7)
            ,spacing=(3.299999952316284,0.78125, 0.78125)
            ,feature_size=24
            ,depths=(2,2,2,2)
            ,is_lucid=True
            ,window_size=(7,7,7)
            ,hparams_dict=self.hparams_dict
            # ,is_deformable=True
            )




        if(self.is_swin_monai):
            img_size=(64,img_size[1],img_size[2])
            # attn_masks_h5f=h5py.File(attn_masks_h5f_path,'w') 
            self.network=SwinUNETR(in_channels=self.num_input_channels
            # ,num_heads=  (1, 3, 6, 12)
             ,num_heads=  (6, 12, 24, 48)
            ,out_channels=self.label_manager.num_segmentation_heads
            ,use_v2=True#
            ,img_size=img_size
            ,patch_size=(2,2,2)
            ,batch_size=self.batch_size
            ,attn_masks_h5f=[]
            ,is_swin=False
            ,is_local_iso=False
            ,is_local_non_iso=False
            # ,distances=(8,8,16)
            ,distances=(7,7,7)
            ,spacing=(3.299999952316284,0.78125, 0.78125)
            ,feature_size=48
            ,depths=(2,2,2,2)
            ,is_lucid=True
            ,window_size=(7,7,7)
            ,use_checkpoint=True
            # ,is_deformable=True
            )


        if self._do_i_compile():
            self.print_to_log_file('Compiling network...')
            self.network = torch.compile(self.network)


        # self.save_hyperparameters()


        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        # if ddp, wrap in DDP wrapper
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])

        self.pl_model= Pl_main_model(network=self.network
                                ,dataloader_train=self.dataloader_train
                                ,dataloader_val=self.dataloader_val
                                ,loss=self.loss
                                ,learning_rate=self.learning_rate
                                ,weight_decay=self.weight_decay
                                ,label_manager=self.label_manager
                                ,log_every_n=self.log_every_n
                                ,num_batch_to_eval=self.num_batch_to_eval
                                ,train_eval_folder=train_eval_folder 
                                ,val_eval_folder=val_eval_folder
                                ,hf5_path=self.hf5_path
                                ,for_explore=for_explore
                                ,batch_size=self.batch_size
                                ,is_classic_nnunet=self.is_classic_nnunet
                                ,is_swin=self.is_swin
                                ,is_med_next=self.is_med_next
                                ,is_swin_monai=self.is_swin_monai
                                ,is_deep_supervision=self.is_deep_supervision
                               ,is_anatomy_segm=self.is_anatomy_segm
                               ,is_lesion_segm=self.is_lesion_segm
                               ,hparams_dict=self.hparams_dict
                                )


        comet_logger = CometLogger(
            api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
            #workspace="OPI", # Optional
            project_name=os.getenv('my_proj_name'), # Optional
            #experiment_name="baseline" # Optional
        )
        
        toMonitor="is_correct_val" 
        mode="max"
        if(self.is_anatomy_segm):
            toMonitor="avgHausdorff_all_val" 
            mode="min"

        checkpoint_callback = ModelCheckpoint(dirpath= self.output_folder,mode=mode, save_top_k=1, monitor=toMonitor)

        # stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=trial.suggest_float("swa_lrs", 1e-6, 1e-4))
        stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=self.learning_rate/2)
        # optuna_prune=PyTorchLightningPruningCallback(trial, monitor=toMonitor)     
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(
            monitor=toMonitor,
            patience=15,
            mode=mode,
            #divergence_threshold=(-0.1)
        )
        # amp_plug=pl.pytorch.plugins.precision.MixedPrecisionPlugin()
        self.trainer = pl.Trainer(
            #accelerator="cpu", #TODO(remove)
            max_epochs=3000,
            # precision='16-mixed', 
            callbacks=[checkpoint_callback], # stochasticAveraging ,stochasticAveraging ,  FineTuneLearningRateFinder(milestones=(5, 10,40)),stochasticAveraging ,FineTuneLearningRateFinder(milestones=(5, 10,40)) early_stopping early_stopping   stochasticAveraging,optuna_prune,checkpoint_callback
            logger=comet_logger,
            accelerator='auto',
            devices='auto',       
            default_root_dir= self.default_root_dir,
            # auto_scale_batch_size="binsearch",
            check_val_every_n_epoch=self.log_every_n,
            accumulate_grad_batches= 12,
            gradient_clip_val = 5.0 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
            log_every_n_steps=self.log_every_n,
            # ,strategy=DDPStrategy(find_unused_parameters=True)
                        # ,reload_dataloaders_every_n_epochs=1
            strategy="deepspeed_stage_1"#_offload
        )
    # def set_deep_supervision_enabled(self, enabled: bool):
    #     """
    #     This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
    #     chances you need to change this as well!
    #     """
    #     if(self.is_classic_nnunet):
    #         if self.is_ddp:
    #             self.network.module.decoder.deep_supervision = enabled
    #         else:
    #             self.network.decoder.deep_supervision = enabled
        

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler

    def _build_loss_lesions(self):
        

        if(self.is_priming_segm):
            loss= FocalLossV2_orig()
        else:
            loss=Picai_FL_and_CE_loss()


        if(self.is_deep_supervision):
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

    def get_training_transforms(self,patch_size: Union[np.ndarray, Tuple[int]],
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

        if(self.is_lesion_segm):
            return self.get_training_transforms_lesions(patch_size,rotation_for_DA,deep_supervision_scales,mirror_axes,do_dummy_2d_data_aug,order_resampling_data
                                                        ,order_resampling_seg,border_val_seg,use_mask_for_norm,is_cascaded,foreground_labels,regions,ignore_label)
        if(self.is_anatomy_segm):
            return self.get_training_transforms_anatomy(patch_size,rotation_for_DA,deep_supervision_scales,mirror_axes,do_dummy_2d_data_aug,order_resampling_data
                                                        ,order_resampling_seg,border_val_seg,use_mask_for_norm,is_cascaded,foreground_labels,regions,ignore_label)


    def get_training_transforms_lesions(self,patch_size: Union[np.ndarray, Tuple[int]],
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
        if(self.is_priming_segm):
            tr_transforms.append(My_priming_setter())
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



    def get_training_transforms_anatomy(self,patch_size: Union[np.ndarray, Tuple[int]],
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
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
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
        # tuner.lr_find(self.pl_model, attr_name="learning_rate")       
        deep_speed_ckpt='/home/sliceruser/nnUNet_results/Dataset294_Prostate/My_Anatomy_trainer__nnUNetPlans__3d_lowres/fold_0/epoch=44-step=900.ckpt'
        self.trainer.fit(self.pl_model)#  , ckpt_path=deep_speed_ckpt , ckpt_path='/home/sliceruser/nnUNet_results/Dataset294_Prostate/My_Anatomy_trainer__nnUNetPlans__3d_lowres/fold_0/epoch=14-step=375.ckpt')
        
        
        
        self.on_train_end()
        # shutil.rmtree(self.default_root_dir)
        # self.f.close()


#my_proj_name="seg anatomy" tag="l4b swin" my_proj_desc=" l4b swin" nnUNetv2_train 294 3d_lowres 0 -tr My_Anatomy_trainer
