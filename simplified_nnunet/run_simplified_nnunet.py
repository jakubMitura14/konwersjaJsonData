import os
from typing import List
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
import pandas as pd
from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import multiprocessing
import os
from multiprocessing import Pool
from typing import List
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from typing import Union, Tuple, List
import einops
import pandas as pd
import h5py
from .dataset import nnUNetDataset_custom





def set_env_variables_for_swin():
    ###os.environ['attn_num_mem_kv'] = '0'
    os.environ['use_scalenorm'] = '0'
    os.environ['sandwich_norm'] = '0'
    os.environ['ff_swish'] = '0'
    os.environ['ff_relu_squared'] = '0'
    os.environ['attn_sparse_topk'] = '0'
    os.environ['attn_talking_heads'] = '0'
    os.environ['attn_on_attn'] = '0'
    os.environ['attn_gate_values'] = '0'
    os.environ['sandwich_coef'] = '0'
    os.environ['macaron'] = '0'
    ###os.environ['residual_attn'] = '0'
    os.environ['gate_residual'] = '0'
    os.environ['shift_tokens'] = '0'
    os.environ['resi_dual'] = '0'
    os.environ['attn_head_scale'] = '0'
    os.environ['ff_post_act_ln'] = '0'
    os.environ['scale_residual'] = '0'
    os.environ['attn_qk_norm'] = '0'
    os.environ['encoders_depth'] = '0'
    os.environ['num_memory_tokens'] = '0'
    os.environ['shift_mem_down'] = '0'



    ### os.environ['attn_num_mem_kv'] = '1'
    # os.environ['use_scalenorm'] = '1'
    # os.environ['sandwich_norm'] = '1'
    # os.environ['ff_swish'] = '1'
    # os.environ['ff_relu_squared'] = '1'
    # os.environ['attn_sparse_topk'] = '1'
    # os.environ['attn_talking_heads'] = '1'
    # os.environ['attn_on_attn'] = '1'
    # os.environ['attn_gate_values'] = '1'
    # os.environ['sandwich_coef'] = '1'
    # os.environ['macaron'] = '1'
    ### os.environ['residual_attn'] = '1'
    # os.environ['gate_residual'] = '1'
    # os.environ['shift_tokens'] = '1'
    # os.environ['resi_dual'] = '1'
    # os.environ['attn_head_scale'] = '1'
    # os.environ['ff_post_act_ln'] = '1'
    # os.environ['scale_residual'] = '1'
    # os.environ['attn_qk_norm'] = '1'
    os.environ['encoders_depth'] = '3'

    ### os.environ['num_memory_tokens'] = '1'
    ### os.environ['shift_mem_down'] = '1'


def set_env_variables_for_general_transforms():
    """
    set the hyperparameters for transforms in lesion segmentation
    """

    # alpha_low=trial.suggest_float("alpha_low", 0.0, 1000.0)
    # os.environ['alpha_low'] = str(alpha_low)
    # os.environ['alpha_high'] = str(alpha_low+trial.suggest_float("alpha_high", 0.0, 2000.0))

    # # sigma_low=trial.suggest_float("sigma_low", 0.0, 19.0)
    # # os.environ['sigma_low'] = str(sigma_low)
    # # os.environ['sigma_high'] = str(sigma_low+trial.suggest_float("sigma_high", 0.0, 20.0))

    # os.environ['GaussianBlurTransform'] = str(trial.suggest_float("GaussianBlurTransform", 0.0,0.9))

    # os.environ['SimulateLowResolutionTransform'] = str(trial.suggest_float("SimulateLowResolutionTransform", 0.0,0.9))
    # os.environ['GammaTransform_a'] = str(trial.suggest_float("GammaTransform_a", 0.0,0.9))
    # os.environ['GammaTransform_b'] = str(trial.suggest_float("GammaTransform_b", 0.0,0.9))
    # os.environ['p_scale_per_sample'] = str(trial.suggest_float("p_scale_per_sample", 0.0,0.9))

    # scale_low=trial.suggest_float("scale_low", 0.0,1.0)
    os.environ['scale_low'] = str(0.1)
    os.environ['scale_high'] = str(0.2)



    # os.environ['p_rot_per_sample'] = str(trial.suggest_float("p_rot_per_sample", 0.0,0.9))
    baseLr=0.00831
    os.environ['learning_rate'] = str(baseLr)#str(trial.suggest_float("learning_rate", baseLr/50,baseLr*2))



    ###os.environ['attn_num_mem_kv'] = '0'
    os.environ['alpha_low'] = str(421.60033013863716)
    os.environ['alpha_high'] = str(421.60033013863716+1451.4488917182302)

    os.environ['sigma_low'] = str(11.541639008878207)
    os.environ['sigma_high'] = str(11.541639008878207+ 9.669272682121036)

    os.environ['p_rot_per_axis']="1"#trial.suggest_categorical("p_rot_per_axis", ["1", "2","3"])
    os.environ['independent_scale_for_each_axis']="1"#trial.suggest_categorical("independent_scale_for_each_axis", ["1","0" ])
    
    # gmma_bottom=trial.suggest_float("gamma_bottom_c", 0.0,3.0)
    # os.environ['gamma_bottom_c']=str(gmma_bottom)
    # os.environ['gamma_up_c']=str(trial.suggest_float("gamma_up_c",  0.0,3.0)+gmma_bottom)
    # os.environ['gamma_prob_c']=str(trial.suggest_float("gamma_prob_c", 0.0,1.0))

    os.environ['RicianNoiseTransform'] = str(0.40230497711508434)
    os.environ['GaussianBlurTransform'] = str(0.30684212800169935)

    os.environ['ContrastAugmentationTransform'] = str(0.6110392464376544)
    os.environ['SimulateLowResolutionTransform'] = str(0.51587313094672182)
    os.environ['GammaTransform_a'] = str(0.09474295989278318)
    os.environ['GammaTransform_b'] = str(0.3249770010008122)
    os.environ['p_el_per_sample'] = str(0.3338220211800589)
    os.environ['p_scale_per_sample'] = str(0.6832566131656115)
    os.environ['p_rot_per_sample'] = str(0.5495580611394884)
    # baseLr=0.00831
    # os.environ['learning_rate'] = str(0.03142343750219331)

def set_norm_and_bias_field():
    # os.environ['to_include_normalize'] = trial.suggest_categorical("to_include_normalize", ["t2w_adc_hbv", "t2w_adc","t2w_hbv","t2w"])
    
    # tau=f"{trial.suggest_float('tau',5e-7,5e-2)}"
    # print(f"tttttttttt {tau}")
    # os.environ['tau'] = tau
    # os.environ['n_classes'] = f"{trial.suggest_int( 'n_classes', 1,20)}"
    # os.environ['log_initialize'] = trial.suggest_categorical("log_initialize", ["0", "1"])
    
    os.environ['to_include_normalize'] = "t2w_adc_hbv"



    os.environ['tau'] = str(0.0426040566281932)
    os.environ['n_classes'] = str(12)
    os.environ['log_initialize'] = "0"
    os.environ['best_metric'] ='0.0'

def setup_pseudo_lesion_adder_and_loss():
    # os.environ['n_lesions'] = str(6)#str(trial.suggest_int("n_lesions", 2,9))
    # os.environ['k_lesions'] = str(200)#str(trial.suggest_int("k_lesions", 0,1000))
    
    os.environ['n_lesions'] = str(7)#str(trial.suggest_int("n_lesions", 3,9))
    os.environ['k_lesions'] = str(30)#str(trial.suggest_int("k_lesions", 1,1000))
    
    # os.environ['output_0_w'] =str(trial.suggest_float("output_0_w", 0.0,1.0))
    # os.environ['output_1_w'] = str(trial.suggest_float("output_1_w", 0.0,1.0))
    os.environ['mean_2'] = str(0.45)#str(trial.suggest_float("mean_2", 0.001,0.999))
    os.environ['mean_3'] = str(3.1)#str(trial.suggest_float("mean_3", 0.001,0.999))
    os.environ['std_0'] = str(0.15)#str(trial.suggest_float("std_0", 0.001,0.999))
    os.environ['std_1'] = str(0.3)#str(trial.suggest_float("std_1", 0.001,0.999))
    os.environ['std_2'] = str(0.15)#str(trial.suggest_float("std_2", 0.001,0.999))
    os.environ['std_3'] = str(0.3)#str(trial.suggest_float("std_3", 0.001,0.999))
    os.environ['mult_old_a'] = str(1.0)#str(trial.suggest_float("mult_old_a", 0.0,1.0))
    os.environ['mult_old_b'] = str(1.0)#str(trial.suggest_float("mult_old_b", 0.0,1.0))
    os.environ['is_anatomic'] = "1"#trial.suggest_categorical("is_anatomic", ["0", "1"])


set_norm_and_bias_field()
setup_pseudo_lesion_adder_and_loss()
set_env_variables_for_general_transforms()




dataset_ids=np.array([4,17,27])
f=h5py.File('/workspaces/konwersjaJsonData/data/hdf5_data.h5', 'r')
dataset_json_file='/workspaces/konwersjaJsonData/simplified_nnunet/jsons_from_nnunet/dataset.json'
plans_file="/workspaces/konwersjaJsonData/simplified_nnunet/jsons_from_nnunet/nnUNetPlans.json"

plans = load_json(plans_file)
input_channels=5
dataset_json = load_json(dataset_json_file)
dataset=nnUNetDataset_custom(hdf5_file=f
                        ,dataset_ids=dataset_ids
                     ,plans_file=plans
                     ,dataset_json=dataset_json
                     ,input_channels=input_channels)


dataset[1]
#python3 -m simplified_nnunet.run_simplified_nnunet
