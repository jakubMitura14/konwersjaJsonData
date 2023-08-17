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
                 ,label_manager):
        super().__init__()
        self.network=network
        self.dataloader_train=dataloader_train
        self.dataloader_val=dataloader_val
        self.loss=loss
        self.learning_rate = learning_rate
        self.weight_decay=weight_decay
        self.label_manager=label_manager
        self.validation_step_outputs = []


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


    def training_step(self, batch, batch_idx):
        self.network.train()
        data = batch['data']
        target = batch['target']
        output = self.network(data)
        return self.loss(output, target)
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
            del data
            l = loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if label_manager.has_ignore_label:
            if not label_manager.has_regions:
                mask = (target != label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        
        percent_in,percent_out,percent_covered,is_correct,my_sensitivity,my_specificity=calc_custom_metrics(epoch,target,output )

        self.validation_step_outputs.append(('loss', l.detach().cpu().numpy()))
        # self.validation_step_outputs.append(('tp_hard',tp_hard))
        # self.validation_step_outputs.append(('fp_hard',fp_hard))
        # self.validation_step_outputs.append(('fn_hard',fn_hard))
        self.validation_step_outputs.append(('percent_in',np.nanmean(percent_in)))
        self.validation_step_outputs.append(('percent_out',np.nanmean(percent_out)))
        self.validation_step_outputs.append(('percent_covered',np.nanmean(percent_covered)))
        self.validation_step_outputs.append(('is_correct',np.nanmean(is_correct)))
        self.validation_step_outputs.append(('my_sensitivity',np.nanmean(my_sensitivity)))
        self.validation_step_outputs.append(('my_specificity',np.nanmean(my_specificity)))

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
        self.log(metr_name, np.nanmean(all.flatten()))

    def on_validation_epoch_end(self):
        # outputs=itertools.chain(*self.validation_step_outputs)
        outputs=self.validation_step_outputs
        
        print(f"validation_epoch_end {outputs}")#{outputs}
        list(map(lambda metr_name : self.parse_outputs(metr_name,outputs),
                 ['loss','percent_in','percent_out','percent_covered','is_correct','my_sensitivity','my_specificity'] ))#'tp_hard','fp_hard','fn_hard'
        
        self.validation_step_outputs.clear()
        # allDices = np.array(([x['dices'] for x in outputs])).flatten() 
        # allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'] for x in outputs])).flatten() 
        # allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'] for x in outputs])).flatten() 
        # allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'] for x in outputs])).flatten() 
        # allaccuracy = np.array(([x['f1_scoree'] for x in outputs])).flatten() 
        
    
        # # allDices = np.array(([x['dices'].cpu().detach().numpy() for x in outputs])).flatten() 
        # # allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'].cpu().detach().numpy() for x in outputs])).flatten() 
        # # allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'].cpu().detach().numpy() for x in outputs])).flatten() 
        # # allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'].cpu().detach().numpy() for x in outputs])).flatten() 
        # regressionMetric=self.regressionMetric.compute()
        # self.regressionMetric.reset()
        # self.log('regr_F1', regressionMetric)
        
        
        # if(len(allDices)>0):            
        #     meanPiecaiMetr_auroc=np.nanmean(allmeanPiecaiMetr_auroc)
        #     meanPiecaiMetr_AP=np.nanmean(allmeanPiecaiMetr_AP)
        #     meanPiecaiMetr_score= np.nanmean(allmeanPiecaiMetr_score)
        #     accuracy= np.nanmean(allaccuracy)
        #     meanPiecaiMetr_score_my= (meanPiecaiMetr_auroc+meanPiecaiMetr_AP+accuracy)/3 #np.nanmean(allmeanPiecaiMetr_score)

        #     self.log('dice', np.nanmean(allDices))

        #     self.log('val_mean_auroc', meanPiecaiMetr_auroc)
        #     self.log('val_mean_AP', meanPiecaiMetr_AP)
        #     self.log('meanPiecaiMetr_score', meanPiecaiMetr_score)
        #     self.log('accuracy', accuracy)
            
        #     self.log('score_my', meanPiecaiMetr_score_my)

        #     self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
        #     self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
        #     self.picaiLossArr_score_final.append(meanPiecaiMetr_score)
        #     self.dice_final.append(np.nanmean(allDices))
