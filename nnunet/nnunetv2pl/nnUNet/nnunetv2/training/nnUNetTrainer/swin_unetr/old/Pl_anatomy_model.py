import monai
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import math
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass
import itertools
# from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
#                                        label_structure, parse_detection_map)
# from picai_eval.image_utils import (read_label, read_prediction,
#                                     resize_image_with_crop_or_pad)
# from picai_eval.metrics import Metrics

# from picai_eval.eval import evaluate_case
import time
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from monai.utils import set_determinism
import math
import torch
from torch.utils.data import random_split, DataLoader
import monai
import pandas as pd
import torchio as tio
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from datetime import datetime
import os
import tempfile
from glob import glob
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from monai.networks.layers.factories import Act, Norm
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import functools
import multiprocessing as mp
import os
import os.path
import shutil
import h5py

from mpi4py import MPI
monai.utils.set_determinism()
from functools import partial
from lightning.pytorch.loggers import CometLogger
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import autocast, nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from ...custom_eval import *
import ignite
import deepspeed

class Pl_anatomy_model(pl.LightningModule):
    def __init__(self,network
                 ,dataloader_train
                 ,dataloader_val
                 ,loss
                 ,learning_rate
                 ,weight_decay
                 ,label_manager
                 ,log_every_n
                 ,num_batch_to_eval
                 ,train_eval_folder
                 ,val_eval_folder
                 ,hf5_path
                 ,for_explore
                 ,batch_size
                 ,is_classic_nnunet
                 ,is_swin
                 ,is_med_next
                 ,is_swin_monai
                 ,is_deep_supervision
                 ,is_anatomy_segm
                 ,is_lesion_segm
                 ):
        

        
        super().__init__()
        self.network=network
        self.dataloader_train=dataloader_train
        self.dataloader_val=dataloader_val
        self.loss=loss
        self.learning_rate = learning_rate
        self.weight_decay=weight_decay
        self.label_manager=label_manager
        self.log_every_n=log_every_n
        self.num_batch_to_eval=num_batch_to_eval
        self.train_eval_folder =train_eval_folder
        self.val_eval_folder =val_eval_folder
        self.hf5_path=hf5_path
        self.for_explore=for_explore
        self.batch_size=batch_size

        self.is_classic_nnunet=is_classic_nnunet
        self.is_swin=is_swin
        self.is_med_next=is_med_next
        self.is_swin_monai=is_swin_monai
        self.is_deep_supervision=is_deep_supervision

        self.is_anatomy_segm=is_anatomy_segm
        self.is_lesion_segm=is_lesion_segm


    def setup(self, stage=None):
        self.logger.experiment.log_text(os.getenv('my_proj_desc'))
        self.logger.experiment.add_tag(os.getenv('tag'))
        self.f = h5py.File(self.hf5_path, 'w',driver='mpio', comm=MPI.COMM_WORLD)
        self.save_hyperparameters()


    def train_dataloader(self):
        return self.dataloader_train                    

    def val_dataloader(self):
        return self.dataloader_val


    def configure_optimizers(self):
        if(self.is_classic_nnunet):
            optimizer = torch.optim.SGD(self.network.parameters(), self.learning_rate, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            # optimizer =deepspeed.ops.adam.DeepSpeedCPUAdam(self.network.parameters(), self.learning_rate)
            
        elif(self.is_swin or self.is_swin_monai):    
            # optimizer = torch.optim.AdamW(self.network.parameters(), 0.07585775750291836)#learning rate set by learning rate finder
            optimizer = deepspeed.ops.adam.FusedAdam(self.network.parameters(), 0.076)#learning rate set by learning rate finder

            # optimizer = deepspeed.ops.adam.FusedAdam(self.network.parameters(), 0.07585775750291836)#learning rate set by learning rate finder


            # optimizer = torch.optim.AdamW(self.network.parameters(), 0.00001)#learning rate set by learning rate finder
            
            # optimizer =deepspeed.ops.adam.DeepSpeedCPUAdam(self.network.parameters(), 0.07585775750291836)
        elif(self.is_med_next):    
            optimizer = torch.optim.AdamW(self.network.parameters(), 0.0019054607179632484)
        
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
        if(self.is_swin or self.is_swin_monai):
            # lr_scheduler = ignite.handlers.param_scheduler.create_lr_scheduler_with_warmup(lr_scheduler, warmup_start_value=0.07585775750291836*40, warmup_duration=30)
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.2, total_iters=30)
            lr_scheduler =torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[scheduler1,lr_scheduler], milestones=[30])
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]


    def on_validation_epoch_start(self):
        self.network.eval()

    def on_train_epoch_start(self):
        #recreating to remove all data
        self.network.train()

    def pad_data_if_needed(self,arr):
        if(self.is_swin_monai):
            return F.pad(arr, (0,0, 0,0, 8,8, 0,0 ,0,0), "constant", 0)
        return arr
    def pad_target_if_needed(self,arr):
        if(self.is_swin_monai):
            if(self.is_deep_supervision):
                return list(map( lambda in_arr :F.pad(in_arr, (0,0, 0,0, 8,8, 0,0 ,0,0), "constant", 0),arr))
            else :
                return F.pad(arr[0], (0,0, 0,0, 8,8, 0,0 ,0,0), "constant", 0)
        return arr
            



    def training_step(self, batch, batch_idx):
        
        data = self.pad_data_if_needed(batch['data'])
        target = self.pad_target_if_needed(batch['target'])
        clinical = torch.tensor(batch['clinical']).to("cuda").float()
        network=self.network
        if(self.is_swin or self.is_swin_monai):
            output = network(data,clinical)
        else:
            output = network(data)        
        # if(not self.is_classic_nnunet):
        #     target=self.transform_gold(target)

        epoch=self.current_epoch
        l=self.loss(output, target)
        print(f"loss {l.detach().cpu().item()}")
        self.log("train loss",l.detach().cpu().item())
        if(epoch%self.log_every_n==0):
            if(batch_idx<self.num_batch_to_eval):
                save_for_metrics(epoch,target,output,data,self.log_every_n,batch_idx,self.f,"train",True)   
        return l


    def transform_gold(self,target):
        shape_0 =target[0].shape
        seg_shape=shape_0

        shape_1 =(seg_shape[2]//2,seg_shape[3]//2,seg_shape[4]//2)
        shape_2 =(shape_1[0]//2,shape_1[1]//2,shape_1[2]//2)
        shape_3 =(shape_2[0]//2,shape_2[1]//2,shape_2[2]//2)
        shape_4 =(shape_3[0]//2,shape_3[1]//2,shape_3[2]//2)
        shape_5 =(shape_4[0]//2,shape_4[1]//2,shape_4[2]//2)

        targets=[target[0]]
        shapes=[shape_0,shape_1,shape_2,shape_3,shape_4,shape_5]     
        for j in range(len(target)):
            if(j>0):
                loc_res=torch.nn.functional.interpolate(input=target[j],size=shapes[j])
                targets.append(loc_res)
        return targets

    def validation_step(self, batch, batch_idx):

        device=self.device
        network=self.network
        loss=self.loss
        label_manager=self.label_manager

        data = self.pad_data_if_needed(batch['data'])
        target = self.pad_target_if_needed(batch['target'])
        clinical = torch.tensor(batch['clinical']).to("cuda").float()


        # if(not self.is_classic_nnunet):
        #     target=self.transform_gold(target)
        
        
        epoch=self.current_epoch
        data = data.to(device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(device, non_blocking=True) for i in target]
        else:
            target = target.to(device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        # with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
        if(self.is_swin or self.is_swin_monai):
            output = network(data,clinical)
        else:
            output = network(data)

        # print(f"ooooooo max {output[0].max()} min {output[0].min()}")
        # del data

        l = loss(output, target)
        save_for_metrics(epoch,target,output,data,self.log_every_n,batch_idx,self.f,"val",True)
        self.log("val loss",l.detach().cpu().item())
        # we only need the output with the highest output resolution
        # output = output[0]
        # target = target[0]
        # if(epoch%self.log_every_n==0):
            # if(batch_idx<self.num_batch_to_eval):
            
        return l
    def my_anato_log(self,tupl,name): 
        print(f"ttt {tupl} {name}")       
        if(np.isnan(tupl[1])):
            self.log(f"{tupl[0]}_{name}", 100.0)
        self.log(f"{tupl[0]}_{name}", tupl[1])
        if(tupl[0]=="avgHausdorff_all" ):
            return True
        
        return False
        
    def on_validation_epoch_end(self):
        group_name='val'
        res= calc_custom_metrics(group_name,self.f,self.for_explore,True,anatomy_metr=True,batch_size=self.batch_size )
        
        main_to_monitor="avgHausdorff_all_val"
        is_there=list(map(lambda tupl : self.my_anato_log(tupl,'val') ,res ))
        
        if(np.sum(np.array(is_there))==0):
             self.log(main_to_monitor, 100.0)
             
    def on_train_epoch_end(self):
        if(self.current_epoch%self.log_every_n==0):
            group_name='train'
            res= calc_custom_metrics(group_name,self.f,self.for_explore,False,anatomy_metr=True,batch_size=self.batch_size )
            list(map(lambda tupl : self.my_anato_log(tupl,'train') ,res ))



