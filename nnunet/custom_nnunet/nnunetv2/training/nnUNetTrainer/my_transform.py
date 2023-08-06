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
from focal_loss.focal_loss import FocalLoss
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.local_transforms import *
from abc import ABC
from typing import Tuple

import numpy as np
import scipy.stats as st
from batchgenerators.utilities.custom_types import ScalarType, sample_scalar
from scipy.ndimage import gaussian_filter
        # tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

class My_BrightnessGradientAdditiveTransform(LocalTransform):
    def __init__(self,
                 scale: ScalarType,
                 loc: ScalarType = (-1, 2),
                 max_strength: ScalarType = 1.,
                 same_for_all_channels: bool = True,
                 mean_centered: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 clip_intensities: bool = False,
                 data_key: str = "data"):
        super().__init__(scale, loc)
        self.max_strength = max_strength
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.same_for_all_channels = same_for_all_channels
        self.mean_centered = mean_centered
        self.clip_intensities = clip_intensities

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        target= data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)
                    if self.mean_centered:
                        # first center the mean of the kernel
                        kernel -= kernel.mean()
                    mx = max(np.max(np.abs(kernel)), 1e-8)
                    if not callable(self.max_strength):
                        strength = sample_scalar(self.max_strength, None, None)
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            # now rescale so that the maximum value of the kernel is max_strength
                            strength = sample_scalar(self.max_strength, data[bi, ci], kernel) if callable(
                                self.max_strength) else strength
                            kernel_scaled = np.copy(kernel) / mx * strength
                            data[bi, ci] += kernel_scaled
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            if self.mean_centered:
                                kernel -= kernel.mean()
                            mx = max(np.max(np.abs(kernel)), 1e-8)
                            strength = sample_scalar(self.max_strength, data[bi, ci], kernel)
                            kernel = kernel / mx * strength
                            data[bi, ci] += kernel
        return data_dict

