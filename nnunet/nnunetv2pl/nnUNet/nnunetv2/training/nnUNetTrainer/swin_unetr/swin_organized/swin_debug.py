import h5py
from .SwinUNETR import *

attn_masks_h5f_path="/workspaces/konwersjaJsonData/sparse_dat/sparse_masks.hdf5"

attn_masks_h5f=h5py.File(attn_masks_h5f_path,'r') 
network=SwinUNETR(in_channels=3
        ,num_heads= (3, 6, 12, 24)
        ,out_channels=3
        ,use_v2=True#
        ,img_size=(64, 64, 64)
        ,patch_size=(1,1,1)
        ,batch_size=1
        ,attn_masks_h5f=attn_masks_h5f
        ,is_swin=False
        ,is_local_iso=True
        ,is_local_non_iso=False
        ,distances=(8,8,8,8)#(4,4,4,4)
        ,spacing=(3.299999952316284,0.78125, 0.78125)
        ,feature_size=24
        ,window_size=4
        ,shift_size=2
        ,depths=(2,2,2,2)
        ).to(device='cuda')

attn_masks_h5f.close()

network(torch.ones((1,3,64, 64, 64)).float().to(device='cuda'),torch.ones((1,3)).float().to(device='cuda') )



# python3 -m nnunet.nnunetv2pl.nnUNet.nnunetv2.training.nnUNetTrainer.swin_unetr.swin_organized.swin_debug

#