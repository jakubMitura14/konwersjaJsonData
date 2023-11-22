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
import torch
from mpi4py import MPI
from lightning.pytorch.loggers import CometLogger


from .data_manag.dataset import nnUNetDataset_custom
from .data_manag.path_manag import manage_paths
from .data_manag.pl_data_module import Main_DataModule
from .model.architecture.select_model import select_model
from .model.Main_pl_model import Pl_main_model
from .model.optimazation import get_learning_rate
from .model.custom_loss import build_loss_function




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


#for manual setup
log_every_n=4
patience=30
num_batch_to_eval=20
batch_size=1
is_deep_supervision=True
is_classic_nnunet=True
is_swin=False
is_swin_monai=False
is_med_next=False
is_lesion_segm=True
is_anatomy_segm= not is_lesion_segm
is_priming_segm= False

#manually specify image sizes and number of input channels
img_size=(48, 192, 160)
if(is_lesion_segm):
    img_size=(40, 96, 96)
input_channels_lesions=5
configuration="3d_lowres"
deep_supervision=True



def run_training(hparams_dict):
    """
    main function for training a model
    """
    # to avoid out of memory    
    torch.cuda.empty_cache()

    #set up paths
    train_eval_folder,val_eval_folder,ligtning_logs_folder,h5_folder,hf5_path,for_explore,source_hdf5_path,dataset_json_file_lesions,plans_file_lesions=manage_paths()

    dataset_ids=np.array([4,17,27])

    source_h5=h5py.File(source_hdf5_path, 'r',driver='mpio', comm=MPI.COMM_WORLD)

    #setup for lesions files that we get from basic nnunet
    plans_lesions = load_json(plans_file_lesions)
    dataset_json_lesions = load_json(dataset_json_file_lesions)
    plans_manager_lesions = PlansManager(plans_lesions)
    configuration_manager_lesions = plans_manager_lesions.get_configuration(configuration)
    label_manager_lesions=plans_manager_lesions.get_label_manager(dataset_json_lesions)

    #setup object for loading and augmenting data
    dataset=nnUNetDataset_custom(hdf5_file=source_h5
                            ,dataset_ids=dataset_ids
                        ,plans_file=plans_manager_lesions
                        ,dataset_json=dataset_json_lesions
                        ,input_channels=input_channels_lesions
                        ,configuration_manager=configuration_manager_lesions)

    #setup neural network for training
    network=select_model(img_size
                    ,is_med_next
                    ,is_swin
                    ,is_swin_monai
                    ,is_classic_nnunet
                    ,dataset_json_lesions
                    ,configuration_manager_lesions
                    ,plans_manager_lesions
                    ,label_manager_lesions
                    ,input_channels_lesions
                    ,batch_size
                    ,deep_supervision)
    #setup pytorch lightning data module
    pl_data_module=Main_DataModule(source_hdf5_file=source_hdf5_path



    #setup main pytorch lightning module
    learning_rate=get_learning_rate(is_swin_monai, is_anatomy_segm, is_classic_nnunet, is_med_next,is_lesion_segm)
    loss=build_loss_function(is_lesion_segm
                            ,is_deep_supervision
                            ,is_anatomy_segm
                            ,configuration_manager_lesions
                            ,label_manager_lesions
                            ,is_priming_segm
                            )
    pl_model= Pl_main_model(network=network
                            ,loss=loss
                            ,learning_rate=learning_rate
                            ,label_manager=label_manager_lesions
                            ,log_every_n=log_every_n
                            ,num_batch_to_eval=num_batch_to_eval
                            ,train_eval_folder=train_eval_folder 
                            ,val_eval_folder=val_eval_folder
                            ,hf5_path=hf5_path
                            ,for_explore=for_explore
                            ,batch_size=batch_size
                            ,is_classic_nnunet=is_classic_nnunet
                            ,is_swin=is_swin
                            ,is_med_next=is_med_next
                            ,is_swin_monai=is_swin_monai
                            ,is_deep_supervision=is_deep_supervision
                            ,is_anatomy_segm=is_anatomy_segm
                            ,is_lesion_segm=is_lesion_segm
                            ,hparams_dict=hparams_dict
                            )

    #logging 
    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name=os.getenv('my_proj_name'), # Optional
        #experiment_name="baseline" # Optional
    )

    toMonitor="is_correct_val" 
    mode="max"
    if(is_anatomy_segm):
        toMonitor="avgHausdorff_pz_val" 
        mode="min"

    checkpoint_callback = ModelCheckpoint(dirpath= output_folder,mode=mode, save_top_k=1, monitor=toMonitor)

    # stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=trial.suggest_float("swa_lrs", 1e-6, 1e-4))
    stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=learning_rate/2)
    # optuna_prune=PyTorchLightningPruningCallback(trial, monitor=toMonitor)     
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor=toMonitor,
        patience=patience,
        mode=mode,
        #divergence_threshold=(-0.1)
    )

    # amp_plug=pl.pytorch.plugins.precision.MixedPrecisionPlugin()
    trainer = pl.Trainer(
        #accelerator="cpu", #TODO(remove)
        max_epochs=1300,
        #gpus=1,
        # precision='16-mixed', 
        callbacks=[checkpoint_callback,stochasticAveraging,early_stopping], # stochasticAveraging ,stochasticAveraging ,  FineTuneLearningRateFinder(milestones=(5, 10,40)),stochasticAveraging ,FineTuneLearningRateFinder(milestones=(5, 10,40)) early_stopping early_stopping   stochasticAveraging,optuna_prune,checkpoint_callback
        logger=comet_logger,
        accelerator='auto',
        devices='auto',       
        default_root_dir= default_root_dir,
        # auto_scale_batch_size="binsearch",
        check_val_every_n_epoch=log_every_n,
        accumulate_grad_batches= 12,
        gradient_clip_val = 5.0 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
        log_every_n_steps=log_every_n
        # ,strategy="ddp_spawn"#DDPStrategy(find_unused_parameters=True)
                    # ,reload_dataloaders_every_n_epochs=1
        # strategy="deepspeed_stage_1"#_offload
    )


    on_train_start()

    if(os.getenv('load_checkpoint')=="1"):
        print(f"loading from checkpoint")
        trainer.fit(pl_model, ckpt_path=os.getenv('checkPoint_path'))
    else:  
        trainer.fit(pl_model)     
            


    on_train_end()