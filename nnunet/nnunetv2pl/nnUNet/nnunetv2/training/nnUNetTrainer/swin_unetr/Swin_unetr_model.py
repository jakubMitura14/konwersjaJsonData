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
from monai.networks.blocks import *
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from xformers.factory import xFormer, xFormerConfig
from xformers.helpers.hierarchical_configs import (
    BasicLayerConfig,
    get_hierarchical_configuration,
)
import xformers.components.attention.attention_patterns as AP
from xformers.components.attention.core import scaled_dot_product_attention
from xformers.components.attention._sputnik_sparse import SparseCS
import xformers.components.attention.attention_patterns as AP
import scipy
import einops
from monai.networks.layers.factories import Act, Norm

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinUNETR",
    # "window_partition",
    # "window_reverse",
    # "WindowAttention",
    # "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]



def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

def get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer):
    p= patch_size
    img_size=img_size
    res=[batch_size,embed_dim,img_size[0]/p[0],img_size[1]/p[1],img_size[2]/p[2]]
    for i in range(i_layer):
        res[1]=res[1]*2
        res[2]=int(res[2]//2)
        res[3]=int(res[3]//2)
        res[4]=int(res[4]//2)
    return res


def get_convs(spatial_dims,patch_size,img_size,batch_size,embed_dim,i_layer,beg_in_channels,norm_name,out_channels):
    """
    we need convolutions in encoding then 
    """
    img_size=get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer)
    # if(i_layer>0):
    #     img_size_b=get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer-1)
    #     beg_in_channels=img_size_b[1]

    encoder = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=img_size[1],
            out_channels=img_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
    decoder = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=img_size[1],
            out_channels=img_size[1]//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
    out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=img_size[1]//2, out_channels=out_channels)
    return (encoder,decoder,out)
    

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
        depths: Sequence[int] = (2, 2, 2, 2),
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
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        # patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        # for m, p in zip(img_size, patch_size):
        #     for i in range(5):
        #         if m % np.power(p, i + 1) != 0:
        #             raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

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
        )
        convsss=list(map(lambda i: get_convs(spatial_dims,patch_size,img_size,batch_size,feature_size,i,in_channels,norm_name,out_channels),range(4)))
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
            feature_size//2,
            kernel_size=3,
            stride=(1,1,1),
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=False,
        )

        # self.decoder_0=UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=img_size[1],
        #     out_channels=int(feature_size*1.5),
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        # self.decoders[0]=self.decoder_0
                # #dec4 torch.Size([5, 384, 3, 3, 3]) dec3 torch.Size([5, 192, 6, 6, 6]) dec2 torch.Size([5, 96, 12, 12, 12]) dec1 torch.Size([5, 48, 24, 24, 24]) dec0 torch.Size([5, 24, 48, 48, 48]) out torch.Size([5, 24, 96, 96, 96])

    def forward(self, x_in):
        # print(f"fffffirst x_in {x_in.shape}")
        hidden_states_out = self.swinViT(x_in, self.normalize)
        # print(f"hhhhhhhhh 0 {hidden_states_out[0].shape} a {hidden_states_out[1].shape} b  {hidden_states_out[2].shape}  c {hidden_states_out[3].shape}")
        enc0 = self.encoder_0(x_in)
        enc1 = self.encoders[0].to('cuda')(hidden_states_out[0])
        enc2 = self.encoders[1].to('cuda')(hidden_states_out[1])
        enc3 = self.encoders[2].to('cuda')(hidden_states_out[2])
        enc4 = self.encoders[3].to('cuda')(hidden_states_out[3])
        print(f"eeeeee {enc4.shape}  enc3 {enc3.shape}")
        dec4= self.decoders[3].to('cuda')(enc4,enc3)
        dec3= self.decoders[2].to('cuda')(dec4,enc2)
        # print(f"yyyyyyyyyyyy dec4 {dec4.shape} enc2 {enc2.shape}")
        dec2= self.decoders[1].to('cuda')(dec3,enc1)
        dec2=self.adaptor(dec2)
        enc0=self.adaptorB(enc0)
        # print(f"iiiiiiiiiiiiiiii dec2 {dec2.shape}  enc0 {enc0.shape}")
        dec1= self.decoders[0].to('cuda')(dec2,enc0)

        # bigger= hidden_states_out[3].shape
        # smaller= dec4.shape
        # # print(f"uuuuuuuuuu bigger {bigger} smaller {smaller}")
        # hidden_states_out[3] = F.pad(hidden_states_out[3], (0,0,0, 0, smaller[2]*2-bigger[2], 0, 0, 0,0, 0))
        # dec3 = self.decoder5(dec4, hidden_states_out[3])
        # if(dec3.shape[2]==4):
        #     dec3=dec3[:,:,1:,:,:]
        # dec2 = self.decoder4(dec3, hidden_states_out[2])
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # print(f"ddddd {dec0.shape} enc0 {enc0.shape}  ")
        # out = self.decoder1(dec0, enc0)

        # print(f"bbbb dec3 {dec3.shape} dec2 {dec2.shape} dec1 {dec1.shape} dec0 {dec0.shape}")

        # print(f" dec4 {dec4.shape} dec3 {dec3.shape} dec2 {dec2.shape} dec1 {dec1.shape} dec0 {dec0.shape} out {out.shape}")
        # #dec4 torch.Size([5, 384, 3, 3, 3]) dec3 torch.Size([5, 192, 6, 6, 6]) dec2 torch.Size([5, 96, 12, 12, 12]) dec1 torch.Size([5, 48, 24, 24, 24]) dec0 torch.Size([5, 24, 48, 48, 48]) out torch.Size([5, 24, 96, 96, 96])
        # logits = self.out(out)
        return [self.outs[0].to('cuda')(dec1),self.outs[1].to('cuda')(dec2),self.outs[2].to('cuda')(dec3),self.outs[3].to('cuda')(dec4)]


class WindowAttention_position_embedding(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__
        
        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, attn,n):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        return attn


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def _generate_nd_grid(*sizes):
    coords = [torch.arange(s) for s in sizes]
    return torch.meshgrid(*coords)


def local_nd_distance(*sizes, p=2.0, weights=None):
    if weights is None:
        weights = (1,) * len(sizes)
    assert len(sizes) == len(weights)
    grid = _generate_nd_grid(*sizes)
    grid = [i.flatten() * w for i, w in zip(grid, weights)]
    grid = torch.stack(grid, dim=1).float()
    d = torch.cdist(grid, grid, p=p)
    return d

def local_nd_pattern(*sizes, distance, p=2.0):
    d = local_nd_distance(*sizes, p=p)
    # print(d)
    return d < distance

class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        img_size=(0,0),
        patch_size=(2,2),
        embed_dim=2,
        i_layer=0
        ,batch_size=1
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_heads=num_heads
        self.dim=dim
        self.img_size=img_size
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.i_layer=i_layer
        self.batch_size=batch_size
        calced_input_size=get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer)
        self.calced_input_size=calced_input_size
        in_attention_distance=10-self.i_layer
        # print(f"sssssssssssss {calced_input_size}")
        pat=local_nd_pattern(calced_input_size[2],calced_input_size[3],calced_input_size[4],distance=in_attention_distance)
        self.attn_mask = SparseCS(pat, torch.device("cuda"))

        proj_drop=0.0
        self.scale = num_heads**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)


        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

 
        # self.embed_dim=embed_dim
        # self.i_layer=i_layer


    def forward(self, x):
        x_shape=x_prim = x.size() 
        b, c, d, h, w = x_shape
        # print(f"pppkkk {x_shape} self.calced_input_size {self.calced_input_size}")
        x= einops.rearrange( x,'b c d h w ->b (d h w) c '
                                ,d=int(self.calced_input_size[2])
                                ,h=int(self.calced_input_size[3])
                                ,w=int(self.calced_input_size[4])
                                )
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        qkv = qkv.flatten(1, 2)

        q, k, v = qkv.unbind()
        
        # x = scaled_dot_product_attention(q, k, v, self.attn_mask.to(device='cuda'), dropout=self.attn_drop)
        # with torch.autocast('cuda', enabled=True):
        out =  self.attn_mask._mat.to('cuda')
        attn_mask=type( self.attn_mask)._wrap(out)
    
        x = scaled_dot_product_attention(q.to(device='cuda'), k.to(device='cuda'), v.to(device='cuda'), attn_mask, dropout=self.attn_drop.to(device='cuda'))        
        x = x.reshape(B, self.num_heads, N, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x= einops.rearrange( x,'b (d h w) c->b d h w c' 
                                ,d=int(self.calced_input_size[2])
                                ,h=int(self.calced_input_size[3])
                                ,w=int(self.calced_input_size[4]) )

        # b, d, h, w,c
        # b, n, c = x.shape
        # qkv = self.qkv(x)
        # qkv_prim_shape=qkv.shape


        # qkv=qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # # if mask is not None:
        #     # print(f"yyyyyyyyyyyyyyy {x.shape}  mask {mask.shape} qkv_prim_shape {qkv_prim_shape} qkv {qkv.shape} \n")
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)

        # attn = self.softmax(attn)

        # attn = self.attn_drop(attn).to(v.dtype)
        # x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        # x = self.proj(x)
        # x = self.proj_drop(x)


        # window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        # x = rearrange(x, "b c d h w -> b d h w c")
        # dp = int(np.ceil(d / window_size[0])) * window_size[0]
        # hp = int(np.ceil(h / window_size[1])) * window_size[1]
        # wp = int(np.ceil(w / window_size[2])) * window_size[2]
        # attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        # for blk in self.blocks:
        #     x = blk(x, attn_mask)
        # print(f"iiiiiiiiiiiiiiiii x {x.shape} x_prim {x_prim} calced {self.get_image_size()}    heads {self.num_heads} dim {self.dim} ")    

        # x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        batch_size:int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = True,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
        img_size=(0)
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.use_v2 = use_v2
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                i_layer=i_layer,
                batch_size=batch_size
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if i_layer == 0:
                    self.layers1c.append(layerc)
                elif i_layer == 1:
                    self.layers2c.append(layerc)
                elif i_layer == 2:
                    self.layers3c.append(layerc)
                elif i_layer == 3:
                    self.layers4c.append(layerc)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x


    

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)

        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        # print(f"000000  x0_out {x0_out.shape}")

        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        # print(f"000000  x1 {x1.shape}")

        x1_out = self.proj_out(x1, normalize)
        # print(f"000000  x1_out {x1_out.shape}")

        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        # print(f"000000  x2 {x1.shape}")

        x2_out = self.proj_out(x2, normalize)
        # print(f"000000  x2_out {x2_out.shape}")

        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        # x4 = self.layers4[0](x3.contiguous())
        # x4_out = self.proj_out(x4, normalize)
        # aaa=1/0
        return [x0_out, x1_out, x2_out, x3_out]#x4_out



network=SwinUNETR(in_channels=3
                                   ,num_heads= (6, 6, 6, 6)
                                #    ,num_heads= (6, 12, 12, 24)

                        ,out_channels=3
                        ,use_v2=True#
                        ,img_size=(48,48,48)
                        ,patch_size=(2,2,2)
                        ,batch_size=1).to(device='cuda')

network(torch.ones((1,3,48,48,48)).float().to(device='cuda'))