from datetime import datetime
from typing import Optional

import datasets
import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
## based on https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_lightning.pytorch.ipynb
## on 3d available https://huggingface.co/maryann-gitonga/brain-tumor-segmentation-3d-attention-unet
