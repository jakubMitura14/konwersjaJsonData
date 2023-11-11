import h5py
from .SwinUNETR import *

attn_masks_h5f_path="/workspaces/konwersjaJsonData/sparse_dat/sparse_masks.hdf5"

# attn_masks_h5f=h5py.File(attn_masks_h5f_path,'r') 
network=SwinUNETR(in_channels=3
        # ,num_heads= (1, 1, 1, 1)
        # ,num_heads= (3, 6, 12, 24)
        ,out_channels=3
        # ,use_v2=True#
        ,img_size=(64, 192, 160)
        # ,patch_size=(2,2,2)
        # ,batch_size=1
        # ,attn_masks_h5f=[]
        # ,is_swin=False
        # ,is_local_iso=False
        # ,is_local_non_iso=False
        # # ,distances=(8,8,16)
        # ,distances=(7,7,7)
        # ,spacing=(3.299999952316284,0.78125, 0.78125)
        # ,feature_size=24
        # ,depths=(2,2,2,2)
        # ,is_lucid=True
        # ,window_size=(7,7,7)
        # ,use_checkpoint=True

             ,num_heads=  (6, 12, 24, 48)
            ,use_v2=True#
            ,patch_size=(2,2,2)
            ,batch_size=1
            ,attn_masks_h5f=[]
            ,is_swin=False
            ,is_local_iso=False
            ,is_local_non_iso=False
            # ,distances=(8,8,16)
            ,distances=(7,7,7)
            ,spacing=(3.299999952316284,0.78125, 0.78125)
            ,feature_size=48
            ,depths=(2,2,2,2)
            ,is_lucid=True
            ,window_size=(7,7,7)
            ,use_checkpoint=True

        # ,downsample="simplified_no_merge"
        ).to(device='cuda')

# attn_masks_h5f.close()

# rr=network(torch.ones((1,3,48, 192, 160)).float().to(device='cuda'),torch.ones((1,3)).float().to(device='cuda') )
rr=network(torch.ones((1,3,64, 192, 160)).float().to(device='cuda'),torch.ones((1,3)).float().to(device='cuda') )
for i,el in enumerate(rr):
        print(f"rr {i}   {el.shape}")


# python3 -m nnunet.nnunetv2pl.nnUNet.nnunetv2.training.nnUNetTrainer.swin_unetr.swin_organized.swin_debug

#

# ** window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([1008, 343, 48]) mul 16595712
# aaaaaaaaaaaa window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([1008, 343, 48]) torch.Size([1008, 343, 48]) attn_mask torch.Size([1008, 1, 343, 343]) dim 48 
# ** window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([1008, 343, 48]) mul 16595712
# ** window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([126, 343, 96]) mul 4148928
# aaaaaaaaaaaa window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([126, 343, 96]) torch.Size([126, 343, 96]) attn_mask torch.Size([126, 1, 343, 343]) dim 96 
# ** window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([126, 343, 96]) mul 4148928
# ** window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([24, 343, 192]) mul 1580544
# aaaaaaaaaaaa window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([24, 343, 192]) torch.Size([24, 343, 192]) attn_mask torch.Size([24, 1, 343, 343]) dim 192 
# ** window_size_corr (7, 7, 7) mul 343  x_windows torch.Size([24, 343, 192]) mul 1580544
# ** window_size_corr (4, 6, 6) mul 144  x_windows torch.Size([8, 144, 384]) mul 442368
# aaaaaaaaaaaa window_size_corr (4, 6, 6) mul 144  x_windows torch.Size([8, 144, 384]) torch.Size([8, 144, 384]) attn_mask torch.Size([4, 1, 245, 245]) dim 384 
# ** window_size_corr (4, 6, 6) mul 144  x_windows torch.Size([8, 144, 384]) mul 442368