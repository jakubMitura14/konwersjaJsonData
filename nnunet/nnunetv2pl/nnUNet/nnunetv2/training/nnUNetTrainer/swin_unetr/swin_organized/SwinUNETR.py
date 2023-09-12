# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")
from .BasicLayer import *
from .swin_utils import*
from .patch_merging import*
from .Window_Attention import*
from .SwinTransformer import*



class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        batch_size:int,
        depths: Sequence[int] = (2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
        patch_size=(2,2,2)
        ,attn_masks_h5f=""
        ,is_swin=False
        ,is_local_iso=False
        ,is_local_non_iso=False
        ,distances=(10,10,10)
        ,spacing=(1.0,1.0,1.0)
        ,window_size=4
        ,shift_size=2
        ,is_deformable=False
        ,is_lucid=False
    ) -> None:

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = patch_size
        # window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(4):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        # if feature_size % 12 != 0:
        #     raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
            img_size=img_size,
            batch_size=batch_size
            ,attn_masks_h5f=attn_masks_h5f
            ,is_swin=is_swin
            ,is_local_iso=is_local_iso
            ,is_local_non_iso=is_local_non_iso
            ,distances=distances
            ,spacing=spacing
            ,shift_size=shift_size
            ,is_deformable=is_deformable
            ,is_lucid=is_lucid
        )

        convsss=list(map(lambda i: get_convs(spatial_dims,patch_size,img_size,batch_size,feature_size,i,in_channels,norm_name,out_channels),range(5)))
        self.encoders= list(map(lambda tupl:tupl[0]  ,convsss))
        self.decoders= list(map(lambda tupl:tupl[1]  ,convsss))
        self.outs= list(map(lambda tupl:tupl[2]  ,convsss))
        self.encoder_0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.adaptor= self.transp_conv = get_conv_layer(
            spatial_dims,
            int(feature_size),
            feature_size,
            kernel_size=3,
            stride=(1,1,1),
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.adaptorB= self.transp_conv = get_conv_layer(
            spatial_dims,
            int(feature_size),
            feature_size,
            kernel_size=3,
            stride=(1,1,1),
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=False,
        )
        self.decoders[0]=None
        self.outs[0]=UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in,clinical):
        hidden_states_out = self.swinViT(x_in, clinical)
        # print(f"x_in {x_in.shape} \n hidden_states_out [0] {hidden_states_out[0].shape} 1) {hidden_states_out[1].shape} 2) {hidden_states_out[2].shape} 3) {hidden_states_out[3].shape} 4) {hidden_states_out[4].shape}" )

        enc0 = self.encoder_0(x_in)
        enc1 = self.encoders[0].to('cuda')(hidden_states_out[0])
        enc2 = self.encoders[1].to('cuda')(hidden_states_out[1])
        enc3 = self.encoders[2].to('cuda')(hidden_states_out[2])
        enc4 = self.encoders[3].to('cuda')(hidden_states_out[3])
        enc5 = self.encoders[4].to('cuda')(hidden_states_out[4])
        dec5= self.decoders[4].to('cuda')(enc5,enc4)
        dec4= self.decoders[3].to('cuda')(dec5,enc3)
        dec3= self.decoders[2].to('cuda')(dec4,enc2)
        dec2= self.decoders[1].to('cuda')(dec3,enc1)

        # dec2=self.adaptor(dec2)
        # enc0=self.adaptorB(enc0)
        dec1= dec2+enc0#self.decoders[0].to('cuda')(dec2,enc0)


        return [self.outs[0].to('cuda')(dec1)+self.outs[1].to('cuda')(dec2)
                ,self.outs[2].to('cuda')(dec3)
                ,self.outs[3].to('cuda')(dec4)
                ,self.outs[4].to('cuda')(dec5)]








