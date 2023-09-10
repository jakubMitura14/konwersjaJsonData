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
import torch.nn.functional as F

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
import xformers.ops as xops

from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig
from xformers.factory.model_factory import xFormer, xFormerConfig
from xformers.components import MultiHeadDispatch, build_attention
from xformers.sparse import BlockSparseTensor, SparseCSRTensor

from xformers.components.feedforward.fused_mlp import FusedMLP
import xformers

from xformers.components.positional_embedding.param import *
from xformers.components.attention.core import _matmul_with_mask
from xformers.components.attention.core import _broadcast_batch
from xformers.components.attention.core import bmm

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

def load_sparse_sputnik(h5f,group_name):
    group_to_load=h5f[group_name]
    values=torch.tensor(group_to_load['values'][()])
    shape=tuple(group_to_load['shape'][()])
    # shape=(values.shape[0],) + shape

    # attn_mask = SparseCSRTensor._wrap((values.shape[0],) + shape
    return xformers.components.attention._sputnik_sparse.SparseCS.wrap(shape=shape
    , values= values
    , row_indices= torch.tensor(group_to_load['row_indices'][()]).type(torch.int32)
    , row_offsets= torch.tensor(group_to_load['row_offsets'][()]).type(torch.int32)
    , column_indices= torch.tensor(group_to_load['column_indices'][()]).type(torch.int32)
    , _transp_info=(torch.tensor(group_to_load['0_transp_info'][()]).type(torch.int32)
                    ,torch.tensor(group_to_load['1_transp_info'][()]).type(torch.int32)
                    ,torch.tensor(group_to_load['2_transp_info'][()]).type(torch.int32)
                    ,torch.tensor(group_to_load['3_transp_info'][()]).type(torch.int32)
                    )
    )


def load_attn_mask_from_h5(h5f,is_swin,is_local_iso,is_local_non_iso ,window_size ,distance , img_size_curr,spacing ):
    
    group_name=f"{int(img_size_curr[0])}_{int(img_size_curr[1])}_{int(img_size_curr[2])}"
    print(f"ggg group_name {group_name}")
    if(is_swin):
        group_name=f"{group_name}/swin/window_{window_size}/main"
        return load_sparse_sputnik(h5f,group_name)
    if(is_local_iso):
        group_name=f"{group_name}/dist_{distance}/iso_vol"
        return load_sparse_sputnik(h5f,group_name)       
    if(is_local_non_iso):
        group_name=f"{group_name}/dist_{distance}_spacing_{spacing[0]}_{spacing[1]}_{spacing[2]}/non_iso_vol"
        return load_sparse_sputnik(h5f,group_name)   

def load_loc_dist_from_h5(h5f,is_swin,is_local_iso,is_local_non_iso ,window_size ,distance , img_size_curr,spacing ):
    
    group_name=f"{int(img_size_curr[0])}_{int(img_size_curr[1])}_{int(img_size_curr[2])}"
    # print(f"kkkkk keys() {h5f[group_name].keys()}")
    # if(is_swin):
    #     group_name=f"{group_name}/swin/window_{window_size}/main"
    #     return load_sparse_sputnik(h5f,group_name)
    if(is_local_iso):
        group_name=f"{group_name}/dist_{distance}/iso_dist"
        keys=np.array(list(h5f[group_name].keys()))
        if("dist_sparse_indicies" in keys):
            # indicies=torch.tensor(h5f[f"{group_name}/dist_sparse_indicies"][()])
            # values=torch.tensor(h5f[f"{group_name}/dist_sparse_values"][()])
            # print(f"iii indicies {indicies.shape}  values {values.shape} indicies type {indicies.dtype} values type {values.dtype} ")

                                  
            return torch.sparse_coo_tensor(indices=torch.tensor(h5f[f"{group_name}/dist_sparse_indicies"][()] )
                                           ,values=torch.tensor(h5f[f"{group_name}/dist_sparse_values"][()]) )  
        else:
            return torch.tensor(h5f[f"{group_name}/dist_dense"][()]  )
    if(is_local_non_iso):
        keys=np.array(list(h5f[group_name].keys()))
        group_name=f"{group_name}/dist_{distance}_spacing_{spacing[0]}_{spacing[1]}_{spacing[2]}/non_iso_dist"
        if("dist_sparse_indicies" in keys):
            return torch.sparse_coo_tensor(indices=torch.tensor(h5f[f"{group_name}/dist_sparse_indicies"][()] )
                                           ,values=torch.tensor(h5f[f"{group_name}/dist_sparse_values"][()]) )  
        else:
            return torch.tensor(h5f[f"{group_name}/dist_dense"][()])



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
        # window_size = ensure_tuple_rep(7, spatial_dims)

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
        self.normalize=True
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

    def forward(self, x_in,clinical):
        # print(f"fffffirst x_in {x_in.shape}")
        hidden_states_out = self.swinViT(x_in,clinical)
        # print(f"hhhhhhhhh 0 {hidden_states_out[0].shape} a {hidden_states_out[1].shape} b  {hidden_states_out[2].shape}  c {hidden_states_out[3].shape}")
        enc0 = self.encoder_0(x_in)
        enc1 = self.encoders[0].to('cuda')(hidden_states_out[0])
        enc2 = self.encoders[1].to('cuda')(hidden_states_out[1])
        enc3 = self.encoders[2].to('cuda')(hidden_states_out[2])
        enc4 = self.encoders[3].to('cuda')(hidden_states_out[3])
        dec4= self.decoders[3].to('cuda')(enc4,enc3)
        dec3= self.decoders[2].to('cuda')(dec4,enc2)
        # print(f"yyyyyyyyyyyy dec4 {dec4.shape} enc2 {enc2.shape}")
        dec2= self.decoders[1].to('cuda')(dec3,enc1)
        dec2=self.adaptor(dec2)
        enc0=self.adaptorB(enc0)
        # print(f"iiiiiiiiiiiiiiii dec2 {dec2.shape}  enc0 {enc0.shape}")
        dec1= self.decoders[0].to('cuda')(dec2,enc0)


        return [self.outs[0].to('cuda')(dec1),self.outs[1].to('cuda')(dec2),self.outs[2].to('cuda')(dec3),self.outs[3].to('cuda')(dec4)]



class Relative_position_embedding_3d(nn.Module):
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
        curr_img_size: Sequence[int],
    ) -> None:


        super().__init__()
        self.dim = dim
        self.curr_img_size = (int(curr_img_size[0]),int(curr_img_size[1]),int(curr_img_size[2]))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.curr_img_size[0] - 1) * (2 * self.curr_img_size[1] - 1) * (2 * self.curr_img_size[2] - 1),
                num_heads,
            )
        )
        coords_d = torch.arange(self.curr_img_size[0])
        coords_h = torch.arange(self.curr_img_size[1])
        coords_w = torch.arange(self.curr_img_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.curr_img_size[0] - 1
        relative_coords[:, :, 1] += self.curr_img_size[1] - 1
        relative_coords[:, :, 2] += self.curr_img_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.curr_img_size[1] - 1) * (2 * self.curr_img_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.curr_img_size[2] - 1


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



class encoding_func_3D:
    """ 
    copied from https://colab.research.google.com/github/osiriszjq/complex_encoding/blob/main/complex_encoding.ipynb#scrollTo=IOvg8IkuI5u0
    """
    def __init__(self, name, param=None):
        encoder_params = {
            'Linf':5.5,
            'Logf':5.5,
            'RFF':14,
            'Gau4':0.006,
            'Tri4':1.5/128
        }
        params=32
        self.name = name



        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn((int(param[1]/2),3))
            elif name == 'rffb':
                self.b = param[0]
            elif name == 'Linf':
                self.b = torch.linspace(2.**0., 2.**param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'Logf':
                self.b = 2.**torch.linspace(0., param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'Gau':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1)
                self.sig = param[0]
            elif name == 'Tri':
                self.dic = torch.linspace(0., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            else:
                print('Undifined encoding!')
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*np.pi*x)),torch.cos((2.*np.pi*x))),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*np.pi*x) @ self.b.T),torch.cos((2.*np.pi*x) @ self.b.T)),1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'Linf')|(self.name == 'Logf'):
            emb1 = torch.cat((torch.sin((2.*np.pi*x[:,:1]) @ self.b.T),torch.cos((2.*np.pi*x[:,:1]) @ self.b.T)),1)
            emb2 = torch.cat((torch.sin((2.*np.pi*x[:,1:2]) @ self.b.T),torch.cos((2.*np.pi*x[:,1:2]) @ self.b.T)),1)
            emb3 = torch.cat((torch.sin((2.*np.pi*x[:,2:3]) @ self.b.T),torch.cos((2.*np.pi*x[:,2:3]) @ self.b.T)),1)
            emb = torch.cat([emb1,emb2,emb3],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Gau':
            emb1 = (-0.5*(x[:,:1]-self.dic)**2/(self.sig**2)).exp()
            emb2 = (-0.5*(x[:,1:2]-self.dic)**2/(self.sig**2)).exp()
            emb3 = (-0.5*(x[:,2:3]-self.dic)**2/(self.sig**2)).exp()
            emb = torch.cat([emb1,emb2,emb3],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Tri':
            emb1 = (1-(x[:,:1]-self.dic).abs()/self.d)
            emb1 = emb1*(emb1>0)
            emb2 = (1-(x[:,1:2]-self.dic).abs()/self.d)
            emb2 = emb2*(emb2>0)
            emb3 = (1-(x[:,2:3]-self.dic).abs()/self.d)
            emb3 = emb3*(emb3>0)
            emb = torch.cat([emb1,emb2,emb3],1)
            emb = emb/(emb.norm(dim=1).max())
            return emb



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
        use_checkpoint: bool = True,
        img_size=(0,0),
        patch_size=(2,2),
        embed_dim=2,
        i_layer=0
        ,batch_size=1
        ,attn_masks_h5f=""
        ,is_swin=False
        ,is_local_iso=False
        ,is_local_non_iso=False
        ,distances=(10,10,10)
        ,spacing=(1.0,1.0,1.0)
        ,shift_size=2
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
        self.shift_size = shift_size
        # self.no_shift = tuple(0 for i in window_size)
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
        # pat=local_nd_pattern(calced_input_size[2],calced_input_size[3],calced_input_size[4],distance=in_attention_distance)
        # self.attn_mask = SparseCS(pat, torch.device("cuda"))

        proj_drop=0.0
        self.scale = num_heads**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.mlp=FusedMLP(dim_model=dim,activation ="gelu",hidden_layer_multiplier=4,dropout=0.05)
        # self.rel_pos_embedding=Relative_position_embedding_3d(dim,num_heads,(calced_input_size[2],calced_input_size[3],calced_input_size[4]))
        self.norm1=nn.LayerNorm(calced_input_size[1])
        self.norm2=nn.LayerNorm(calced_input_size[1])

        self.clinical_dense=nn.Linear(3,dim)
        self.clinical_MLP=FusedMLP(dim_model=dim,activation ="gelu",hidden_layer_multiplier=1,dropout=0.05)

        

        # self.attn_mask = SparseCSRTensor._wrap(shape, values, row_indices, row_offsets, column_indices, _transp_info)
        self.attn_mask = load_attn_mask_from_h5(h5f=attn_masks_h5f
                                                ,is_swin=is_swin
                                                ,is_local_iso=is_local_iso
                                                ,is_local_non_iso=is_local_non_iso
                                                 ,window_size=window_size
                                                 ,distance=distances[i_layer]
                                                , img_size_curr=(calced_input_size[2],calced_input_size[3],calced_input_size[4])
                                                ,spacing=spacing )

        # self.softmax = nn.Softmax(dim=-1)


        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=3)

        DROPOUT=0.0
        EMB=calced_input_size[1] // self.num_heads
        SEQ=int(calced_input_size[2]*calced_input_size[3]*calced_input_size[4])
        # print(f"ooooo EMB {EMB} SEQ {SEQ}")
        my_config = {
            "name": "scaled_dot_product",  # you can easily make this dependent on a file, sweep,..
            "dropout": DROPOUT,
            "seq_len": SEQ,
            "attention_query_mask": None#torch.rand((SEQ, 1)) < 0.3, # some dummy mask
        }
        # if(self.i_layer==2):
        #     my_config = {
        #         "name": "nystrom",  # you can easily make this dependent on a file, sweep,..
        #         "dropout": DROPOUT,
        #         "seq_len": SEQ,
        #         "num_heads":num_heads,
        #         "attention_query_mask": None#torch.rand((SEQ, 1)) < 0.3, # some dummy mask
        #     }  
        # else:
        self.attn_mask = load_attn_mask_from_h5(h5f=attn_masks_h5f
                                    ,is_swin=is_swin
                                    ,is_local_iso=is_local_iso
                                    ,is_local_non_iso=is_local_non_iso
                                        ,window_size=window_size
                                        ,distance=distances[i_layer]
                                    , img_size_curr=(calced_input_size[2],calced_input_size[3],calced_input_size[4])
                                    ,spacing=spacing )    
        # attention = xops.memory_efficient_attention

        attention=build_attention(my_config)

        # build a multi head dispatch to test this attention mechanism
        self.multi_head = MultiHeadDispatch(
            seq_len=SEQ,
            dim_model=EMB,
            residual_dropout=DROPOUT,
            num_heads=num_heads,
            attention=attention,
        )
        self.new_emb=False
        self.my_simple_rel_emb=False
        self.xformers_relative_embed=True
        if(self.xformers_relative_embed):
            self.pos_emb=LearnablePositionalEmbedding(SEQ,EMB, add_class_token=False)
        if(self.new_emb):
            self.pos_emb=encoding_func_3D('Gau')
        if(self.my_simple_rel_emb):
            self.loc_dists=load_loc_dist_from_h5(attn_masks_h5f,is_swin,is_local_iso,is_local_non_iso ,window_size ,distances[i_layer] , (calced_input_size[2],calced_input_size[3],calced_input_size[4]),spacing )    



    def forward(self, x,clinical):
        x,B, N, C,shortcut= checkpoint.checkpoint(self.forward_main_a,x)
        return checkpoint.checkpoint(self.forward_main_b,x,B, N, C,clinical,shortcut)
    def forward_main_a(self, x):
        x_shape=x_prim = x.size() 
        b, c, d, h, w = x_shape
        # print(f"pppkkk {x_shape} self.calced_input_size {self.calced_input_size}")
        x= einops.rearrange( x,'b c d h w ->b (d h w) c '
                                ,d=int(self.calced_input_size[2])
                                ,h=int(self.calced_input_size[3])
                                ,w=int(self.calced_input_size[4])
                                )
        B, N, C = x.shape
        shortcut=x
        # self.norm_layer0(x)
        qkv=self.qkv(x)

        qkv = (
            qkv
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        qkv = qkv.flatten(1, 2)
        q, k, v = qkv.unbind()


        # x = scaled_dot_product_attention(q, k, v, self.attn_mask.to(device='cuda'), dropout=self.attn_drop)
        # with torch.autocast('cuda', enabled=True):

        # x = scaled_dot_product_attention(q.to(device='cuda'), k.to(device='cuda'), v.to(device='cuda'), attn_mask, dropout=self.attn_drop.to(device='cuda'))        
        
        # if(self.i_layer!=2):
        out =  self.attn_mask._mat.to('cuda')
        attn_mask=type( self.attn_mask)._wrap(out)            
        x=self.multi_head(F.normalize(q, dim=-1), F.normalize(k, dim=-1), v,attn_mask)
        # else:
        #     x=self.multi_head(F.normalize(q, dim=-1), F.normalize(k, dim=-1), v)


        if(self.my_simple_rel_emb):
            loc_dists= torch.cat([self.loc_dists for _ in range(v.shape[0])], dim=0).to('cuda')
            x= x+ torch.bmm(loc_dists,v)

        if(self.new_emb or self.xformers_relative_embed):
            x=self.pos_emb(x)

        # x = xops.memory_efficient_attention(q, k, v, op=None)
        return x,B, N, C,shortcut
        # self.rel_pos_embedding(x,N)
    def forward_main_b(self, x,B, N, C,clinical,shortcut):
    
        x = x.reshape(B, self.num_heads, N, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)+self.clinical_MLP(self.clinical_dense(clinical )) ))


        x= einops.rearrange( x,'b (d h w) c->b d h w c' 
                                ,d=int(self.calced_input_size[2])
                                ,h=int(self.calced_input_size[3])
                                ,w=int(self.calced_input_size[4]) )


                        
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
        ,attn_masks_h5f=""
        ,is_swin=False
        ,is_local_iso=False
        ,is_local_non_iso=False
        ,distances=(10,10,10)
        ,spacing=(1.0,1.0,1.0)
        ,window_size=4
        ,shift_size=2
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
                ,attn_masks_h5f=attn_masks_h5f
                ,is_swin=is_swin
                ,is_local_iso=is_local_iso
                ,is_local_non_iso=is_local_non_iso
                ,distances=distances
                ,spacing=spacing
                ,shift_size=shift_size
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


    

    def forward(self, x,clinical):
        x0 = checkpoint.checkpoint(self.patch_embed,x)
        normalize=True
        x0 = self.pos_drop(x0)
        x0_out = checkpoint.checkpoint(self.proj_out,x0, normalize)

        if self.use_v2:
            x0 = checkpoint.checkpoint(self.layers1c[0],x0.contiguous())
        x1 = checkpoint.checkpoint(self.layers1[0],x0.contiguous(),clinical)

        x1_out = checkpoint.checkpoint(self.proj_out,x1, normalize)

        if self.use_v2:
            x1 = checkpoint.checkpoint(self.layers2c[0],x1.contiguous())
        x2 = checkpoint.checkpoint(self.layers2[0],x1.contiguous(),clinical)

        x2_out = checkpoint.checkpoint(self.proj_out,x2, normalize)

        if self.use_v2:
            x2 = checkpoint.checkpoint(self.layers3c[0],(x2.contiguous()))
        x3 = checkpoint.checkpoint(self.layers3[0],x2.contiguous(),clinical)
        x3_out = checkpoint.checkpoint(self.proj_out,x3, normalize)
        # if self.use_v2:
        #     x3 = self.layers4c[0](x3.contiguous())
        # x4 = self.layers4[0](x3.contiguous())
        # x4_out = self.proj_out(x4, normalize)
        # aaa=1/0
        return [x0_out, x1_out, x2_out, x3_out]#x4_out


import h5py

attn_masks_h5f_path="/workspaces/konwersjaJsonData/sparse_dat/sparse_masks.hdf5"

attn_masks_h5f=h5py.File(attn_masks_h5f_path,'r') 
network=SwinUNETR(in_channels=3
        ,num_heads= (2,8,8)
        ,out_channels=3
        ,use_v2=True#
        ,img_size=(32, 32, 32)
        ,patch_size=(2,2,2)
        ,batch_size=1
        ,attn_masks_h5f=attn_masks_h5f
        ,is_swin=False
        ,is_local_iso=True
        ,is_local_non_iso=False
        ,distances=(8,8,8)#(4,4,4,4)
        ,spacing=(3.299999952316284,0.78125, 0.78125)
        ,feature_size=32
        ,window_size=4
        ,shift_size=2
        ).to(device='cuda')

attn_masks_h5f.close()

network(torch.ones((1,3,32, 32, 32)).float().to(device='cuda'),torch.ones((1,3)).float().to(device='cuda') )
# network(torch.ones((1,3,48, 192, 160)).float().to(device='cuda'))