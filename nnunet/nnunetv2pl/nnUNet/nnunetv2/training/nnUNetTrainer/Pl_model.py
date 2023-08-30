import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

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
import pytorch_lightning as pl
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
from pytorch_lightning.loggers import CometLogger
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import autocast, nn
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from .custom_eval import *


class Pl_Model(pl.LightningModule):
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
                 ,is_plain_conv
                  ,batch_size  
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
        self.is_plain_conv=is_plain_conv
        
        
        self.batch_size=batch_size
                
        # self.validation_step_outputs = []
        # self.test_step_outputs = []



    def setup(self, stage=None):
        self.logger.experiment.log_text(os.getenv('my_proj_desc'))
        self.logger.experiment.add_tag(os.getenv('tag'))
        self.f = h5py.File(self.hf5_path, 'w',driver='mpio', comm=MPI.COMM_WORLD)



    def train_dataloader(self):
        return self.dataloader_train                    

    def val_dataloader(self):
        return self.dataloader_val


    def configure_optimizers(self):

        TODO change
        optimizer = torch.optim.SGD(self.network.parameters(), self.learning_rate/2, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        
        
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        lr_scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        # warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
        #                                 len_loader=len(self.dataloader_train),
        #                                 warmup_steps=30,
        #                                 warmup_start_lr=0.01,
        #                                 warmup_mode='linear')
        
        
        # optimizer = torch.optim.AdamW(self.network.parameters(), 0.003311311214825908)#learning rate set by learning rate finder
        
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
        # lr_scheduler =pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,20,1000) 
        
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]


    # def infer_train_ds_labels(self, batch):
    #     x, y, numLesions = batch["train_ds_labels"]['chan3_col_name'] , batch["train_ds_labels"]['label'], batch["train_ds_labels"]['num_lesions_to_retain']
    #     segmMap,regr = self.modelRegression(x)
    #     return segmMap,regr, y, numLesions


    # def infer_train_ds_no_labels(self, batch):
    #     x, numLesions =batch["train_ds_no_labels"]['chan3_col_name'],batch["train_ds_no_labels"]['num_lesions_to_retain']
    #     segmMap,regr = self.modelRegression(x)
    #     return regr, numLesions


    def on_validation_epoch_start(self):
        self.network.eval()

    def on_train_epoch_start(self):
        #recreating to remove all data
        self.network.train()

    def training_step(self, batch, batch_idx):
        
        data = batch['data']
        target = batch['target']
        output = self.network(data)
        if(not self.is_plain_conv):
            target=self.transform_gold(target)
        # print(f"tttt target {len(target)}   {target[0].shape}")
        
        epoch=self.current_epoch
        l=self.loss(output, target)

        if(epoch%self.log_every_n==0):
            if(batch_idx<self.num_batch_to_eval):
                save_for_metrics(epoch,target,output,data,self.log_every_n,batch_idx,self.f,"train")
                # print(f"bbbbbbbbbbbb {batch_idx} output {len(output)} {output[0].shape} {output[1].shape} {output[2].shape}")
                # percent_in,percent_out,percent_covered,is_correct,my_sensitivity,my_specificity=calc_custom_metrics(epoch,target,output ,self.log_every_n,batch_idx)

                # self.test_step_outputs.append(('loss', l.detach().cpu().numpy()))
                # # self.validation_step_outputs.append(('tp_hard',tp_hard))
                # # self.validation_step_outputs.append(('fp_hard',fp_hard))
                # # self.validation_step_outputs.append(('fn_hard',fn_hard))
                # self.test_step_outputs.append(('percent_in',np.nanmean(percent_in)))
                # self.test_step_outputs.append(('percent_out',np.nanmean(percent_out)))
                # self.test_step_outputs.append(('percent_covered',np.nanmean(percent_covered)))
                # self.test_step_outputs.append(('is_correct',np.nanmean(is_correct)))
                # self.test_step_outputs.append(('my_sensitivity',np.nanmean(my_sensitivity)))
                # self.test_step_outputs.append(('my_specificity',np.nanmean(my_specificity)))
        
        
        
        return l
        # if(self.current_epoch%2):
        #     seg_hat,reg_hat, y_true, numLesions=self.infer_train_ds_labels( batch)
        #     return torch.add(self.criterion(seg_hat,y_true)
        #                     ,self.regLoss(reg_hat.flatten().float(),torch.Tensor(numLesions).to(self.device).flatten().float() ) 
        #                     # ,self.regLoss(regr_no_lab.flatten(),torch.Tensor(numLesions_no_lab).to(self.device).flatten() ) 
        #                         )
        # else:
        #     regr_no_lab, numLesions_no_lab= self.infer_train_ds_no_labels( batch) 
        #     return self.regLoss(regr_no_lab.flatten().float(),torch.Tensor(numLesions_no_lab).to(self.device).flatten().float() ) 


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
        # print(f"ttttttttttt {target[0].shape}")   
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

        data = batch['data']
        target = batch['target']
        if(not self.is_plain_conv):
            target=self.transform_gold(target)
        
        
        # if(isinstance(data,list)):
        #     print(f"valll data 0 {data[0].shape} {data[1].shape} {data[2].shape}")
        # else:
        #     print(f"valll data 0 {data.shape}")


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
        with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            output = network(data)
            TODO check is softmax/sigmoid needed

            # del data
            l = loss(output, target)
            save_for_metrics(epoch,target,output,data,self.log_every_n,batch_idx,self.f,"val")
        self.log("val loss",l.detach().cpu().item())
        # we only need the output with the highest output resolution
        # output = output[0]
        # target = target[0]
        # if(epoch%self.log_every_n==0):
            # if(batch_idx<self.num_batch_to_eval):
            
        return l

    def on_validation_epoch_end(self):
        group_name='val'
        res= calc_custom_metrics(group_name,self.f,self.for_explore,True,batch_size=self.batch_size ).flatten()        
        
        self.log("percent_in_val", res[0]) #,sync_dist=True
        self.log("percent_out_val", res[1]) #,sync_dist=True
        self.log("percent_covered_val", res[2]) #,sync_dist=True
        self.log("is_correct_val", res[3])#,sync_dist=True
        self.log("my_sensitivity_val", res[4])#,sync_dist=True
        self.log("my_specificity_val", res[5])#,sync_dist=True

        self.log("num_components_val", res[6])#,sync_dist=True
        self.log("in_inferred_val", res[7])#,sync_dist=True



        # outputs=self.validation_step_outputs        
        # list(map(lambda metr_name : self.parse_outputs(metr_name,outputs),
        #          ['loss','percent_in','percent_out','percent_covered','is_correct','my_sensitivity','my_specificity'] ))#'tp_hard','fp_hard','fn_hard'
        
        # self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        if(self.current_epoch%self.log_every_n==0):
            group_name='train'
            res= calc_custom_metrics(group_name,self.f,self.for_explore,False,batch_size=self.batch_size ).flatten()
            self.log("percent_in_train", res[0]) #,sync_dist=True
            self.log("percent_out_train", res[1]) #,sync_dist=True
            self.log("percent_covered_train", res[2]) #,sync_dist=True
            self.log("is_correct_train", res[3])#,sync_dist=True
            self.log("my_sensitivity_train", res[4])#,sync_dist=True
            self.log("my_specificity_train", res[5])#,sync_dist=True
            self.log("num_components_train", res[6])#,sync_dist=True
            self.log("in_inferred_train", res[7])#,sync_dist=True



