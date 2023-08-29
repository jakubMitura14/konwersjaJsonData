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
from .Pl_anatomy_model import *
import shutil
import h5py
import pytorch_lightning as pl
from lightning.pytorch.tuner import Tuner
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP



import transformers
from mpi4py import MPI
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers import AutoImageProcessor
from .med_next.create_mednext_v1 import *
from .swin_unetr.Swin_unetr_model import *


class My_Anatomy_trainer(nnUNetTrainer):

    def on_train_start(self):
        """
        we will additionally invoke here the initialization of pytorch lightning module
        """
        self.log_every_n=5
        self.num_batch_to_eval=20

        self.is_classic_nnunet=False
        self.is_swin=True
        self.is_med_next=False
        
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




        self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                self.dataset_json)

        if(self.is_classic_nnunet):
            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                            self.configuration_manager,
                                                            self.num_input_channels,
                                                            enable_deep_supervision=True).to(self.device)
        # compile network for free speedup





        # self.network=create_mednextv1_large(num_input_channels=self.num_input_channels
        #                                     ,num_classes=self.label_manager.num_segmentation_heads
        #                                     ,kernel_size= 3
        #                                     ,ds= True)
        if(self.is_swin):
            self.network=SwinUNETR(in_channels=self.num_input_channels
                        ,out_channels=self.label_manager.num_segmentation_heads
                        ,use_v2=True#
                        ,img_size=(32, 256, 256))

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



        self.pl_model= Pl_anatomy_model(network=self.network
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
                                ,batch_size=self.batch_size
                                ,is_classic_nnunet=self.is_classic_nnunet
                                ,is_swin=self.is_swin
                                ,is_med_next=self.is_med_next)
        


        # print(f"oooooooooooooo {self.output_folder}")
        # self.pl_model=Pl_anatomy_model.load_from_checkpoint(self.output_folder)
        

        comet_logger = CometLogger(
            api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
            #workspace="OPI", # Optional
            project_name=os.getenv('my_proj_name'), # Optional
            #experiment_name="baseline" # Optional
        )


        toMonitor="avgHausdorff_all_val"
        checkpoint_callback = ModelCheckpoint(dirpath= self.output_folder,mode='min', save_top_k=1, monitor=toMonitor)
        # stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=trial.suggest_float("swa_lrs", 1e-6, 1e-4))
        stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=1e-3)
        # optuna_prune=PyTorchLightningPruningCallback(trial, monitor=toMonitor)     
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(
            monitor=toMonitor,
            patience=15,
            mode="min",
            #divergence_threshold=(-0.1)
        )
        # amp_plug=pl.pytorch.plugins.precision.MixedPrecisionPlugin()
        self.trainer = pl.Trainer(
            #accelerator="cpu", #TODO(remove)
            max_epochs=1000,
            #gpus=1,
            #precision=experiment.get_parameter("precision"), 
            callbacks=[checkpoint_callback], # early_stopping early_stopping   stochasticAveraging,optuna_prune,checkpoint_callback
            logger=comet_logger,
            accelerator='auto',
            devices='auto',       
            default_root_dir= self.default_root_dir,
            # auto_scale_batch_size="binsearch",
            check_val_every_n_epoch=self.log_every_n,
            accumulate_grad_batches= 1,
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

    # def _build_loss(self):
    #     loss= FocalLossV2_orig()

    #     deep_supervision_scales = self._get_deep_supervision_scales()

    #     # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    #     # this gives higher resolution outputs more weight in the loss
    #     weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    #     weights[-1] = 0

    #     # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    #     weights = weights / weights.sum()
    #     # now wrap the loss
    #     loss = DeepSupervisionWrapper(loss, weights)
    #     return loss    



    def run_training(self):
        self.on_train_start()

        tuner = Tuner(self.trainer)
        # to set to your own hparams.my_value
        tuner.lr_find(self.pl_model, attr_name="learning_rate")
        self.trainer.fit(self.pl_model)
        
        self.on_train_end()
        # shutil.rmtree(self.default_root_dir)
        # self.f.close()


