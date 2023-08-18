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
                 ,f
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
        self.f=f
        # self.validation_step_outputs = []
        # self.test_step_outputs = []


    def setup(self, stage=None):
        self.logger.experiment.log_text(os.getenv('my_proj_desc'))
        self.logger.experiment.add_tag(os.getenv('tag'))



    def train_dataloader(self):
        return self.dataloader_train                    

    def val_dataloader(self):
        return self.dataloader_val


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.learning_rate, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
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
        self.network.train()

    def training_step(self, batch, batch_idx):
        
        data = batch['data']
        target = batch['target']
        output = self.network(data)

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
        
        self.log("train loss",l.detach().cpu().item())
        
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


    def validation_step(self, batch, batch_idx):
        
        device=self.device
        network=self.network
        loss=self.loss
        label_manager=self.label_manager

        data = batch['data']
        target = batch['target']

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
            # del data
            l = loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]
        if(epoch%self.log_every_n==0):
            save_for_metrics(epoch,target,output,data,self.log_every_n,batch_idx,self.f,"val")
        # # the following is needed for online evaluation. Fake dice (green line)
        # axes = [0] + list(range(2, len(output.shape)))

        # if label_manager.has_regions:
        #     predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        # else:
        #     # no need for softmax
        #     output_seg = output.argmax(1)[:, None]
        #     predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        #     predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        #     del output_seg

        # if label_manager.has_ignore_label:
        #     if not label_manager.has_regions:
        #         mask = (target != label_manager.ignore_label).float()
        #         # CAREFUL that you don't rely on target after this line!
        #         target[target == label_manager.ignore_label] = 0
        #     else:
        #         mask = 1 - target[:, -1:]
        #         # CAREFUL that you don't rely on target after this line!
        #         target = target[:, :-1]
        # else:
        #     mask = None

        # tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        # tp_hard = tp.detach().cpu().numpy()
        # fp_hard = fp.detach().cpu().numpy()
        # fn_hard = fn.detach().cpu().numpy()
        # if not label_manager.has_regions:
        #     # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        #     # (softmax training) there needs tobe one output for the background. We are not interested in the
        #     # background Dice
        #     # [1:] in order to remove background
        #     tp_hard = tp_hard[1:]
        #     fp_hard = fp_hard[1:]
        #     fn_hard = fn_hard[1:]


        # percent_in,percent_out,percent_covered,is_correct,my_sensitivity,my_specificity=calc_custom_metrics(epoch,target,output ,self.log_every_n,batch_idx)

        # self.validation_step_outputs.append(('loss', l.detach().cpu().numpy()))
        # # self.validation_step_outputs.append(('tp_hard',tp_hard))
        # # self.validation_step_outputs.append(('fp_hard',fp_hard))
        # # self.validation_step_outputs.append(('fn_hard',fn_hard))
        # self.validation_step_outputs.append(('percent_in',np.nanmean(percent_in)))
        # self.validation_step_outputs.append(('percent_out',np.nanmean(percent_out)))
        # self.validation_step_outputs.append(('percent_covered',np.nanmean(percent_covered)))
        # self.validation_step_outputs.append(('is_correct',np.nanmean(is_correct)))
        # self.validation_step_outputs.append(('my_sensitivity',np.nanmean(my_sensitivity)))
        # self.validation_step_outputs.append(('my_specificity',np.nanmean(my_specificity)))

        # 'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard
        #                         , 'percent_in' :percent_in,'percent_out':percent_out,'percent_covered':percent_covered,'is_correct':is_correct
        #                         ,'my_sensitivity':my_sensitivity,'my_specificity':my_specificity
        return l
        # return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard
        #                 , 'percent_in' :percent_in,'percent_out':percent_out,'percent_covered':percent_covered,'is_correct':is_correct
        #                 ,'my_sensitivity':my_sensitivity,'my_specificity':my_specificity   }


    def parse_outputs(self,metr_name,outputs ):
        all = np.array(list(filter(lambda t:t[0]== metr_name,outputs)))
        all = np.array(list(map(lambda t:np.array(t[1]).flatten(),outputs)))
        self.log(metr_name, np.nanmean(all.flatten()),sync_dist=True)

    def on_validation_epoch_end(self):
        if(self.current_epoch%self.log_every_n==0):
            group_name='val'
            return calc_custom_metrics(group_name,self.f)

        # outputs=self.validation_step_outputs        
        # list(map(lambda metr_name : self.parse_outputs(metr_name,outputs),
        #          ['loss','percent_in','percent_out','percent_covered','is_correct','my_sensitivity','my_specificity'] ))#'tp_hard','fp_hard','fn_hard'
        
        # self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        if(self.current_epoch%self.log_every_n==0):
            group_name='train'
            return calc_custom_metrics(group_name,self.f)

