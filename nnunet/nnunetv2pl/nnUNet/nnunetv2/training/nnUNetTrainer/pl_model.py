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

from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

from picai_eval.eval import evaluate_case
import time
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from monai.utils import set_determinism
import math
import torch
from torch.utils.data import random_split, DataLoader
import monai
import gdown
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


class Pl_Model(pl.LightningModule):
    def __init__(self,net
                 ,dataloader_train
                 ,dataloader_val
                 ,loss
                 ,learning_rate):
        super().__init__()
        self.net=net
        self.dataloader_train=dataloader_train
        self.dataloader_val=dataloader_val
        self.loss=loss
        self.learning_rate = learning_rate

    def setup(self, stage=None):
        self.experiment.log_text(os.getenv('my_proj_desc'))
        self.experiment.add_tag(os.getenv('tag'))

    def train_dataloader(self):
        return self.dataloader_train                    

    def val_dataloader(self):
        return self.dataloader_val


    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
        return [optimizer], [lr_scheduler]


    def infer_train_ds_labels(self, batch):
        x, y, numLesions = batch["train_ds_labels"]['chan3_col_name'] , batch["train_ds_labels"]['label'], batch["train_ds_labels"]['num_lesions_to_retain']
        segmMap,regr = self.modelRegression(x)
        return segmMap,regr, y, numLesions


    def infer_train_ds_no_labels(self, batch):
        x, numLesions =batch["train_ds_no_labels"]['chan3_col_name'],batch["train_ds_no_labels"]['num_lesions_to_retain']
        segmMap,regr = self.modelRegression(x)
        return regr, numLesions


    def training_step(self, batch, batch_idx):
        if(self.current_epoch%2):
            seg_hat,reg_hat, y_true, numLesions=self.infer_train_ds_labels( batch)
            return torch.add(self.criterion(seg_hat,y_true)
                            ,self.regLoss(reg_hat.flatten().float(),torch.Tensor(numLesions).to(self.device).flatten().float() ) 
                            # ,self.regLoss(regr_no_lab.flatten(),torch.Tensor(numLesions_no_lab).to(self.device).flatten() ) 
                                )
        else:
            regr_no_lab, numLesions_no_lab= self.infer_train_ds_no_labels( batch) 
            return self.regLoss(regr_no_lab.flatten().float(),torch.Tensor(numLesions_no_lab).to(self.device).flatten().float() ) 


    def validation_step(self, batch, batch_idx):
        print("start validation")
        experiment=self.experiment=self.logger.experiment
        #log hyperparameters if it is epoch 1
        if(self.toLogHyperParam):
            self.logHyperparameters(experiment)

        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name_val'], batch['label_name_val'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        numBatches = y_true.size(dim=0)
        #seg_hat, reg_hat = self.modelRegression(x)        
        # seg_hat, reg_hat = self.modelRegression(x)        
        seg_hat,regr = self.modelRegression(x)
        print(f"regr raw {regr}")
        seg_hat = seg_hat.cpu().detach()
        regr=torch.sigmoid(regr)
        print(f"regr sigm  {regr}")
        # self.regressionMetric(regr.flatten().float(),torch.Tensor(numLesions).to(self.device).flatten().float())
        regrr=torch.round(regr.flatten()).int().cpu().detach().numpy()
        numL=torch.Tensor(numLesions).cpu().int().detach().numpy()
        print(f"regr{regrr} numL {numL} ")

        # f1_scoree = sklearn.metrics.accuracy_score(numL,regrr)
        conff=sklearn.metrics.confusion_matrix(numL,regrr).ravel()
       
        f1_scoree=0.0
        # f1_scoree=float(np.array_equal( numL,regrr )) #exactly the same
        # alt= (1-float(np.array_equal( numL,np.logical_not(numL))))
        if(np.array_equal( numL,regrr )):
            f1_scoree=1.0
        elif(np.array_equal( numL,np.logical_not(numL))):
            f1_scoree=0.0
        else:
            tn, fp, fn, tp = conff 
            f1_scoree=(tp+tn)/(tp+fp+fn+tn)


        
        #f1_scoree = sklearn.metrics.balanced_accuracy_score(numL,regrr)
        print(f"loc f1_score {f1_scoree}")
        self.regressionMetric(torch.round(regr.flatten().float()),torch.Tensor(numLesions).to(self.device).float())
        regr=regr.cpu().detach().numpy()
        # regr= list(map(lambda el : int(el>0.5) ,regr ))
        seg_hat=torch.sigmoid(seg_hat).cpu().detach()
        # diceLocRaw=monai.metrics.compute_generalized_dice( self.postProcessA(seg_hat) ,y_true.cpu())[1].cpu().detach().item()

        # t2wb=decollate_batch(batch['t2wb'])
        # labelB=decollate_batch(batch['labelB'])
        #loss= self.criterion(seg_hat,y_true)# self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)      
        y_det = decollate_batch(seg_hat.cpu().detach())
        # y_background = decollate_batch(seg_hat[:,0,:,:,:].cpu().detach())
        y_true = decollate_batch(y_true.cpu().detach())
        patIds = decollate_batch(batch['study_id'])
        numLesions = decollate_batch(batch['num_lesions_to_retain'])
        images = decollate_batch(x.cpu().detach()) 

        # print(f"val num batches {numBatches} t2wb {t2wb} patIds {patIds} labelB {labelB}")
        print(f"val num batches {numBatches} ")
        lenn=numBatches
        processedCases=[]
        my_task=partial(processDecolated,gold_arr=y_true,y_hat_arr=y_det,directory= self.temp_val_dir,studyId= patIds
                    ,imageArr=images, postProcess=self.postProcess,epoch=self.current_epoch,regr=regr,threshold=self.threshold)
        with mp.Pool(processes = mp.cpu_count()) as pool:
            #it = pool.imap(my_task, range(lenn))
            results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
            time.sleep(60)
            processedCases=list(map(lambda ind :getNext(ind,results,self.toWaitForPostProcess) ,list(range(lenn)) ))

        isTaken= list(map(lambda it:type(it) != type(None),processedCases))
        extracteds=list(filter(lambda it:type(it) != type(None),processedCases))

        lenn=len(extracteds)
        print(f"lenn after extract {lenn}")
        # extracteds=list(filter(lambda it:it.numpy(),extracteds))


        # processedCases=list(map(partial(processDecolated,gold_arr=y_true,y_hat_arr=y_det,directory= self.temp_val_dir,studyId= patIds
        #             ,imageArr=images, experiment=self.logger.experiment,postProcess=self.postProcess,epoch=self.current_epoch)
        #             ,range(0,numBatches)))
        # y_detD=list(map(lambda entry : self.postProcess(entry) ,y_det  ))
        # y_detD= torch.stack(y_detD).cpu()
        # goldsFull = torch.stack(y_true).cpu()
        # diceLocRaw=0.0
        # diceLocRaw=monai.metrics.compute_generalized_dice( y_detD.cpu() ,goldsFull)[1].item()
                

        # try:
        #     diceLocRaw=monai.metrics.compute_generalized_dice( y_detD.cpu() ,goldsFull)[1].item()
        # except:
        #     pass  

        if(len(extracteds)>1):
            directory= self.temp_val_dir
            epoch=self.current_epoch
            list(map(partial(log_images
                ,experiment=experiment,golds=y_true,extracteds=extracteds 
                ,t2ws=images,directory=directory ,patIds=patIds,epoch=epoch,numLesions=numLesions),range(lenn)))
            # y_true= list(map(lambda el: el.numpy()  ,y_true))                                              
            meanPiecaiMetr_auroc=0.0
            meanPiecaiMetr_AP=0.0
            meanPiecaiMetr_score= 0.0
            try:
                valid_metrics = evaluate(y_det=extracteds,
                                        y_true=list(map(lambda el: el.numpy()[1,:,:,:]  ,y_true)),
                                        num_parallel_calls= os.cpu_count()
                                        ,verbose=1)
                meanPiecaiMetr_auroc=0.0 if math.isnan(valid_metrics.auroc) else valid_metrics.auroc
                meanPiecaiMetr_AP=0.0 if math.isnan(valid_metrics.AP) else valid_metrics.AP
                meanPiecaiMetr_score= 0.0 if math.isnan(valid_metrics.score) else  valid_metrics.score
            except:
                pass
            print("start dice")
            extracteds= list(map(lambda numpyEntry : self.postProcess(torch.from_numpy((numpyEntry>0).astype('int8'))) ,extracteds  ))
            extracteds= torch.stack(extracteds)
            



            # extracteds= self.postProcess(extracteds)#argmax=True,
            y_truefil=list(filter(lambda tupl:  isTaken[tupl[0]] , enumerate(y_true)))
            y_truefil=list(map(lambda tupl:  tupl[1] ,y_truefil))
            golds=torch.stack(y_truefil).cpu()

            # print(f"get dice  extrrr {extracteds.cpu()}  Y true  {y_true_prim.cpu()}   ")
            diceLoc=0.0
            # diceLoc=monai.metrics.compute_generalized_dice( extracteds.cpu() ,golds)[1].item()

            try:
                diceLoc=monai.metrics.compute_generalized_dice( extracteds.cpu() ,golds)[1].item()
            except:
                pass    
  


            # print(f"diceLoc {diceLoc} diceLocRaw {diceLocRaw}")

            # gold = list(map(lambda tupl: tupl[2] ,processedCases ))

            return {'dices': diceLoc, 'meanPiecaiMetr_auroc':meanPiecaiMetr_auroc
                    ,'meanPiecaiMetr_AP' :meanPiecaiMetr_AP,'meanPiecaiMetr_score': meanPiecaiMetr_score, 'f1_scoree':f1_scoree}

        return {'dices': 0.0, 'meanPiecaiMetr_auroc':0.0
                ,'meanPiecaiMetr_AP' :0.0,'meanPiecaiMetr_score': 0.0, 'f1_scoree':f1_scoree}




    def validation_epoch_end(self, outputs):
        print("validation_epoch_end")

        allDices = np.array(([x['dices'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'] for x in outputs])).flatten() 
        allaccuracy = np.array(([x['f1_scoree'] for x in outputs])).flatten() 
        
    
        # allDices = np.array(([x['dices'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'].cpu().detach().numpy() for x in outputs])).flatten() 
        regressionMetric=self.regressionMetric.compute()
        self.regressionMetric.reset()
        self.log('regr_F1', regressionMetric)
        
        
        if(len(allDices)>0):            
            meanPiecaiMetr_auroc=np.nanmean(allmeanPiecaiMetr_auroc)
            meanPiecaiMetr_AP=np.nanmean(allmeanPiecaiMetr_AP)
            meanPiecaiMetr_score= np.nanmean(allmeanPiecaiMetr_score)
            accuracy= np.nanmean(allaccuracy)
            meanPiecaiMetr_score_my= (meanPiecaiMetr_auroc+meanPiecaiMetr_AP+accuracy)/3 #np.nanmean(allmeanPiecaiMetr_score)

            self.log('dice', np.nanmean(allDices))

            print(f"accuracy {accuracy} meanPiecaiMetr_score_my {meanPiecaiMetr_score_my} meanDice {np.nanmean(allDices)} regr_F1 {regressionMetric}  meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )
            self.log('val_mean_auroc', meanPiecaiMetr_auroc)
            self.log('val_mean_AP', meanPiecaiMetr_AP)
            self.log('meanPiecaiMetr_score', meanPiecaiMetr_score)
            self.log('accuracy', accuracy)
            
            self.log('score_my', meanPiecaiMetr_score_my)

            self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
            self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
            self.picaiLossArr_score_final.append(meanPiecaiMetr_score)
            self.dice_final.append(np.nanmean(allDices))
