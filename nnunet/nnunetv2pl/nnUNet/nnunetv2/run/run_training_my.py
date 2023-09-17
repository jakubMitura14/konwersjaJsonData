import torch
from .run_training import run_training

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device('cuda')

run_training(dataset_name_or_id='101',
                 configuration='3d_lowres',
                 trainer_class_name = 'Main_trainer_pl',
                 plans_identifier = 'nnUNetPlans'
                 ,fold=0)

# 294 3d_lowres 0

# Main_trainer_pl


# python3 -m nnunet.nnunetv2pl.nnUNet.nnunetv2.run.run_training_my