import torch
from .run_training import run_training

device="cuda"
if device == 'cpu':
    # let's allow torch to use hella threads
    import multiprocessing
    torch.set_num_threads(multiprocessing.cpu_count())
    device = torch.device('cpu')
elif device == 'cuda':
    # multithreading in torch doesn't help nnU-Net if run on GPU
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda')
else:
    device = torch.device('mps')

run_training(dataset_name_or_id='101',
                 configuration='3d_lowres',
                 trainer_class_name = 'Main_trainer_pl',
                 plans_identifier = 'nnUNetPlans'
                 ,fold=0)

# 294 3d_lowres 0

# Main_trainer_pl


# python3 -m nnunet.nnunetv2pl.nnUNet.nnunetv2.run.run_training_my