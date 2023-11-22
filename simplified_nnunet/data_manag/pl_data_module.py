import os
from typing import List
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
import pandas as pd
# from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import multiprocessing
import os
from multiprocessing import Pool
from typing import List
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from typing import Union, Tuple, List
import einops
import pandas as pd
import h5py
from ..dataset import nnUNetDataset_custom
import lightning.pytorch as pl
import lightning as L
from mpi4py import MPI
import os
import torch.utils.data.DataLoader as DataLoader


class Main_DataModule(L.LightningDataModule):
    """
    source_hdf5_file- path to hdf5 with input data
    dataset_json_file- path to dataset.json file that was created using original nnunet
    plans_file- path to plans.json file that was created using original nnunet
    """

    def __init__(self, source_f
                 ,dataset_json_file
                 ,plans_file
                 ,input_channels
                 ,train_dataset_ids
                 ,val_dataset_ids
                 ,batch_size=1):
        super().__init__()
        self.source_f = source_f
        self.dataset_json_file=dataset_json_file
        self.plans_file=plans_file
        self.input_channels=input_channels     
        self.train_dataset_ids=train_dataset_ids   
        self.val_dataset_ids=val_dataset_ids
        self.batch_size=batch_size

    # def prepare_data(self):
    #     # download
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def setup(self):    
        dataset_json = load_json(self.dataset_json_file)
        plans = load_json(self.plans_file)
        #training dataset
        self.train_dataset=nnUNetDataset_custom(hdf5_file=self.source_f
                        ,dataset_ids=self.train_dataset_ids
                     ,plans_file=plans
                     ,dataset_json=dataset_json
                     ,input_channels=self.input_channels)
        #validation dataset
        self.val_dataset=nnUNetDataset_custom(hdf5_file=self.source_f
                        ,dataset_ids=self.val_dataset_ids
                     ,plans_file=plans
                     ,dataset_json=dataset_json
                     ,input_channels=self.input_channels)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,pin_memory=True,num_workers=os.cpu_count,shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,pin_memory=True,num_workers=os.cpu_count)

