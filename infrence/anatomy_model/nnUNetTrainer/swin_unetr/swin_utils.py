
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
