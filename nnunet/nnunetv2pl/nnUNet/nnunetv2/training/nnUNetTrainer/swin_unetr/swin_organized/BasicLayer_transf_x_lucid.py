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

from .swin_utils import*
from .patch_merging import*
from .Window_Attention import*
from .my_x_transformers import *
import os
rearrange, _ = optional_import("einops", name="rearrange")


# class Relative_position_embedding_3d(nn.Module):
#     """
#     Window based multi-head self attention module with relative position bias based on: "Liu et al.,
#     Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
#     <https://arxiv.org/abs/2103.14030>"
#     https://github.com/microsoft/Swin-Transformer
#     """

#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         curr_img_size: Sequence[int],
#     ) -> None:


#         super().__init__()
#         self.dim = dim
#         self.curr_img_size = (int(curr_img_size[0]),int(curr_img_size[1]),int(curr_img_size[2]))
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim**-0.5
#         mesh_args = torch.meshgrid.__kwdefaults__
        
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros(
#                 (2 * self.curr_img_size[0] - 1) * (2 * self.curr_img_size[1] - 1) * (2 * self.curr_img_size[2] - 1),
#                 num_heads,
#             )
#         )
#         coords_d = torch.arange(self.curr_img_size[0])
#         coords_h = torch.arange(self.curr_img_size[1])
#         coords_w = torch.arange(self.curr_img_size[2])
#         if mesh_args is not None:
#             coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
#         else:
#             coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
#         coords_flatten = torch.flatten(coords, 1)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += self.curr_img_size[0] - 1
#         relative_coords[:, :, 1] += self.curr_img_size[1] - 1
#         relative_coords[:, :, 2] += self.curr_img_size[2] - 1
#         relative_coords[:, :, 0] *= (2 * self.curr_img_size[1] - 1) * (2 * self.curr_img_size[2] - 1)
#         relative_coords[:, :, 1] *= 2 * self.curr_img_size[2] - 1


#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
#         trunc_normal_(self.relative_position_bias_table, std=0.02)

#     def forward(self,n):
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
#         ].reshape(n, n, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         return relative_position_bias.unsqueeze(0)




def get_transf_from_hyper(dim,num_heads,window_size_corr,calced_input_size  ):
        #transformer_level
        # attn_flash = True,
        #     ff_glu = True
        kwargss_encoder={'ff_glu':True
                         }

        if(os.getenv('attn_num_mem_kv')=='1'):
            kwargss_encoder["attn_num_mem_kv"] = 16
        if(os.getenv('use_scalenorm')=='1'):
            kwargss_encoder["use_scalenorm"] = True
            kwargss_encoder["sandwich_norm"] = False
        if(os.getenv('sandwich_norm')=='1'):
            kwargss_encoder["use_scalenorm"] = False
            kwargss_encoder["sandwich_norm"] = True
        if(os.getenv('ff_swish')=='1'):
            kwargss_encoder["ff_relu_squared"] = False
            kwargss_encoder["ff_swish"] = True
        if(os.getenv('ff_relu_squared')=='1'):
            kwargss_encoder["ff_relu_squared"] = True
            kwargss_encoder["ff_swish"] = False
        if(os.getenv('attn_sparse_topk')=='1'):
            kwargss_encoder["attn_sparse_topk"] = 12
        if(os.getenv('attn_talking_heads')=='1'):
            kwargss_encoder["attn_talking_heads"] = True
        if(os.getenv('attn_on_attn')=='1'):
            kwargss_encoder["attn_on_attn"] = True
        if(os.getenv('attn_gate_values')=='1'):
            kwargss_encoder["attn_gate_values"] = True
        if(os.getenv('sandwich_coef')=='1'):
            kwargss_encoder["sandwich_coef"] = 6
        if(os.getenv('macaron')=='1'):
            kwargss_encoder["macaron"] = True
        if(os.getenv('residual_attn')=='1'):
            kwargss_encoder["residual_attn"] = True
            kwargss_encoder["pre_norm"] = False
        if(os.getenv('gate_residual')=='1'):
            kwargss_encoder["gate_residual"] = True
        if(os.getenv('shift_tokens')=='1'):
            kwargss_encoder["shift_tokens"] = 1
        if(os.getenv('resi_dual')=='1'):
            kwargss_encoder["resi_dual"] = True
            kwargss_encoder["resi_dual_scale"] = 0.1
        if(os.getenv('attn_head_scale')=='1'):
            kwargss_encoder["attn_head_scale"] = True
        if(os.getenv('ff_post_act_ln')=='1'):
            kwargss_encoder["ff_post_act_ln"] = True
        if(os.getenv('scale_residual')=='1'):
            kwargss_encoder["scale_residual"] = True
        if(os.getenv('attn_qk_norm')=='1'):
            kwargss_encoder["attn_qk_norm"] = True
            kwargss_encoder["attn_qk_norm_groups"] = 16
        if(os.getenv('attn_qk_norm_dim_scale')=='1'):
            kwargss_encoder["attn_qk_norm_dim_scale"] = True

        depth=int(os.getenv('encoders_depth'))
        kwargss_encoder["depth"]=depth
        #performance
        # ff_no_bias = True
        # attn_one_kv_head = True
        # attn_kv_heads = 2

        attn_layers = Encoder_my(
            dim = dim,
            depth = 1,
            heads = num_heads,
            is_Relative_position_embedding_3d=True,
            window_size=window_size_corr,
            calced_input_size=calced_input_size,
            return_hiddens=False,
            **kwargss_encoder
        )


        #for transformer level
        # num_memory_tokens = 20
        # l2norm_embed = True | post_emb_norm = True
        # shift_mem_down = 1,

        if(os.getenv('num_memory_tokens')=='1'):
            attn = My_transformer_wrapper(
                dim_in = dim,
                dim_out = dim,
                max_seq_len = 10024,
                attn_layers=attn_layers,
                num_memory_tokens=20
                )

        if(os.getenv('shift_mem_down')=='1'):
            attn = My_transformer_wrapper(
                dim_in = dim,
                dim_out = dim,
                max_seq_len = 10024,
                attn_layers=attn_layers,
                shift_mem_down=1
                )

        if(os.getenv('num_memory_tokens')=='1' and os.getenv('shift_mem_down')=='1'):
            attn = My_transformer_wrapper(
                dim_in = dim,
                dim_out = dim,
                max_seq_len = 10024,
                attn_layers=attn_layers,
                num_memory_tokens=20,
                shift_mem_down=1
                )
            

        return attn



class SwinTransformerBlock_lucid(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        calced_input_size=(1,1,1,1)
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = True
        self.norm1 = norm_layer(dim)
        self.calced_input_size=calced_input_size
        # self.attn = WindowAttention(
        #     dim,
        #     window_size=self.window_size,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )

        #transformer_level
        # num_memory_tokens = 20
        # l2norm_embed = True | post_emb_norm = True
        # shift_mem_down = 1,

        # #encoder_level
        # attn_num_mem_kv = 16
        # use_scalenorm = True |sandwich_norm = True
        # ff_swish = True | ff_relu_squared = True
        # attn_sparse_topk = 10
        # attn_talking_heads = True
        # attn_on_attn = True
        # attn_gate_values = True
        # sandwich_coef = 6
        # macaron = True
        # residual_attn = True plus pre_norm = False
        # gate_residual = True
        # shift_tokens = 1
        # resi_dual = True plus resi_dual_scale = 0.1
        # attn_head_scale = True
        # ff_post_act_ln = True
        # scale_residual = True
        # attn_qk_norm = True plus attn_qk_norm_groups = 8
        # attn_qk_norm = True
        # attn_qk_norm_dim_scale = True
        # #performance
        # ff_no_bias = True
        # attn_one_kv_head = True
        # attn_kv_heads = 2


        b, d, h, w, c = self.calced_input_size
        window_size_corr, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        self.attn = My_transformer_wrapper(
            dim_in = dim,
            dim_out = dim,
            max_seq_len = 10024,
            attn_layers = Encoder_my(
                dim = dim,
                depth = 2,
                heads = num_heads,
                is_Relative_position_embedding_3d=True,
                window_size=window_size_corr,
                calced_input_size=calced_input_size,
                return_hiddens=False,
                 attn_flash = True,
                ff_glu = True
            )
        )


        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")
        # self.clinical_dense=nn.Linear(3,dim)

    def forward_part1(self, x, mask_matrix,clinical):
        # print(f"ffffffffffff  forward_part1  dim {self.dim} x {x.shape} calced_input_size {self.calced_input_size} ")
        x_shape = x.size()
        # shortcut = x
        # x = self.norm1(x)
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]


        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        if(attn_mask is not None):
            attn_mask= einops.rearrange(attn_mask,'bb a b-> bb 1 a b')
            attn_mask=(attn_mask*(-1))/100
            attn_mask=torch.logical_not(attn_mask.bool())

            print(f"aaaaaaaaaaaa  {x_windows.shape} attn_mask {attn_mask.shape} dim {self.dim} ")
        print(f"** x_windows {x_windows.shape} ")
        attn_windows = self.attn(x_windows, attn_mask=attn_mask,clinical=clinical)

        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()

        return x

    # def forward_part2(self, x,shortcut,clinical):
    #     x = shortcut + self.drop_path(self.norm1(x))
    #     x = x + self.drop_path(self.norm2(self.mlp(x))+self.clinical_dense(clinical ))
    #     return x

    def forward(self, x, mask_matrix,clinical):
        x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix,clinical)
        # x = checkpoint.checkpoint(self.forward_part2, x,shortcut,clinical)

        return x



class BasicLayer_lucid(nn.Module):
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
        self.i_layer=i_layer
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        calced_input_size=get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock_lucid(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    calced_input_size=calced_input_size
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size),)

    def forward(self, x,clinical):
        x_shape = x.size()
        b, c, d, h, w = x_shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask,clinical)
        x = x.view(b, d, h, w, -1)
        
        
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

"""
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

from .swin_utils import*
from .patch_merging import*
from .Window_Attention import*
from .my_x_transformers import *
rearrange, _ = optional_import("einops", name="rearrange")


class Relative_position_embedding_3d(nn.Module):
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

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

    def forward(self,n):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)





class SwinTransformerBlock_lucid(nn.Module):
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        calced_input_size=(1,1,1,1)
    ) -> None:
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = True
        self.norm1 = norm_layer(dim)
        self.calced_input_size=calced_input_size
        # self.attn = WindowAttention(
        #     dim,
        #     window_size=self.window_size,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )
        
        self.attn = My_transformer_wrapper(
            dim_in = dim,
            dim_out = dim,
            max_seq_len = 10024,
            attn_layers = Encoder_my(
                dim = dim,
                depth = 6,
                heads = num_heads,
                is_Relative_position_embedding_3d=True,
                window_size=window_size,
                calced_input_size=calced_input_size,
                return_hiddens=False,
                attn_flash = True
            )
        )


        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")
        self.clinical_dense=nn.Linear(3,dim)

    def forward_part1(self, x, mask_matrix,clinical):
        # print(f"ffffffffffff  forward_part1  dim {self.dim} x {x.shape} calced_input_size {self.calced_input_size} ")
        x_shape = x.size()
        # shortcut = x
        # x = self.norm1(x)
        b, d, h, w, c = x.shape


        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        # pad_l = pad_t = pad_d0 = 0
        # pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        # pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        # pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]


        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        print(f"wwwwwwwwwwwwww shifted_x {shifted_x.shape} window_size {window_size}")
        x_windows = window_partition(shifted_x, window_size)
        if(attn_mask is not None):
            attn_mask= einops.rearrange(attn_mask,'bb a b-> bb 1 a b')
            attn_mask=(attn_mask*(-1))/100
            attn_mask=torch.logical_not(attn_mask.bool())
        print(f"xxfff {x_windows.shape}")
        attn_windows = self.attn(x_windows, attn_mask=attn_mask,clinical=clinical)

        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

        else:
            x = shifted_x

        # if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
        #     x = x[:, :d, :h, :w, :].contiguous()



        return x

    # def forward_part2(self, x,shortcut,clinical):
    #     x = shortcut + self.drop_path(self.norm1(x))
    #     x = x + self.drop_path(self.norm2(self.mlp(x))+self.clinical_dense(clinical ))
    #     return x

    def forward(self, x, mask_matrix,clinical):
        x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix,clinical)
        # x = checkpoint.checkpoint(self.forward_part2, x,shortcut,clinical)

        return x



class BasicLayer_lucid(nn.Module):
  


        super().__init__()
        self.i_layer=i_layer
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        calced_input_size=get_image_size(patch_size,img_size,batch_size,embed_dim,i_layer)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock_lucid(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    calced_input_size=calced_input_size
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size),)

    def forward(self, x,clinical):
        x_shape = x.size()
        print(f"iii {self.i_layer}")
        b, c, d, h, w = x_shape
        window_size=self.window_size

        x = rearrange(x, "b c d h w -> b d h w c")
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        b2, d2, h2, w2, c2 = x_shape

        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)


        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask,clinical)
        x = x.view(b2, d2, h2, w2, -1)
        if self.downsample is not None:
            x = self.downsample(x)

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()

        x = rearrange(x, "b d h w c -> b c d h w")
        return x


"""
