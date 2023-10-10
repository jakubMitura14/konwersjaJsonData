import optuna
import os
from subprocess import Popen
import subprocess
from optuna.storages import RetryFailedTrialCallback
import importlib.util
import importlib
import sys
import pandas as pd
import numpy as np

# def loadLib(name,path):
#     spec = importlib.util.spec_from_file_location(name, path)
#     res = importlib.util.module_from_spec(spec)
#     sys.modules[name] = res
#     spec.loader.exec_module(res)
#     return res

# csv_dir="/workspaces/konwersjaJsonData/hyperopt/curr_csv.csv"
# curr_csv = pd.DataFrame([{"ress":0.0}])

# print(f"rrrrrrrrr {curr_csv}")
# curr_csv.to_csv(csv_dir) 

with open('/workspaces/konwersjaJsonData/hyperopt/curr_npy.npy', 'wb') as f:
    np.save(f, np.array([0]))

# seg_lesions_custom=loadLib("seg_lesions_custom","/workspaces/konwersjaJsonData/nnunet/seg_lesions_custom.py")

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


def set_env_variables_for_general_transforms(trial):
    """
    set the hyperparameters for transforms in lesion segmentation
    """
    ###os.environ['attn_num_mem_kv'] = '0'
    alpha_low=trial.suggest_float("alpha_low", 0.0, 1000.0)
    os.environ['alpha_low'] = str(alpha_low)
    os.environ['alpha_high'] = str(alpha_low+trial.suggest_float("alpha_high", 0.0, 2000.0))

    sigma_low=trial.suggest_float("sigma_low", 0.0, 19.0)
    os.environ['sigma_low'] = str(sigma_low)
    os.environ['sigma_high'] = str(sigma_low+trial.suggest_float("sigma_high", 0.0, 20.0))

    # os.environ['p_rot_per_axis']=trial.suggest_categorical("p_rot_per_axis", ["1", "2","3"])

    os.environ['RicianNoiseTransform'] = str(trial.suggest_float("RicianNoiseTransform", 0.0,0.9))
    os.environ['GaussianBlurTransform'] = str(trial.suggest_float("GaussianBlurTransform", 0.0,0.9))

    os.environ['ContrastAugmentationTransform'] = str(trial.suggest_float("ContrastAugmentationTransform", 0.0,0.9))
    os.environ['SimulateLowResolutionTransform'] = str(trial.suggest_float("SimulateLowResolutionTransform", 0.0,0.9))
    os.environ['GammaTransform_a'] = str(trial.suggest_float("GammaTransform_a", 0.0,0.9))
    os.environ['GammaTransform_b'] = str(trial.suggest_float("GammaTransform_b", 0.0,0.9))
    os.environ['p_el_per_sample'] = str(trial.suggest_float("p_el_per_sample", 0.0,0.9))
    os.environ['p_scale_per_sample'] = str(trial.suggest_float("p_scale_per_sample", 0.0,0.9))
    os.environ['p_rot_per_sample'] = str(trial.suggest_float("p_rot_per_sample", 0.0,0.9))
    baseLr=0.00831
    os.environ['learning_rate'] = str(trial.suggest_float("learning_rate", baseLr/1000,baseLr*10))


def set_norm_and_bias_field(trial):
    os.environ['to_include_normalize'] = trial.suggest_categorical("to_include_normalize", ["t2w_adc_hbv", "t2w_adc","t2w_hbv","t2w"])
    
    tau=f"{trial.suggest_float('tau',5e-7,5e-2)}"
    print(f"tttttttttt {tau}")
    os.environ['tau'] = tau
    os.environ['n_classes'] = f"{trial.suggest_int( 'n_classes', 1,20)}"
    os.environ['log_initialize'] = trial.suggest_categorical("log_initialize", ["0", "1"])

set_env_variables_for_swin()
os.environ['best_metric'] ='0.0'


# experiment_name="general_augment"
experiment_name="classic_augmentations2"#bias_norm


def objective(trial: optuna.trial.Trial) -> float:

    #checking if there is some failed trial if so we will restart it
    expId = RetryFailedTrialCallback.retried_trial_number(trial)
    if(expId is None):
        expId=trial.number

    # set_norm_and_bias_field(trial)
    # seg_lesions_custom.main_func()
    set_env_variables_for_general_transforms(trial)

    cmd=f"my_proj_name='hyperparam_classic_aug' tag='hyperparam_classic_aug' my_proj_desc='hyperparam_classic_aug' nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl"

    # p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
    p = Popen(cmd, shell=True)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE

    p.wait()

    numpy_dir="/workspaces/konwersjaJsonData/hyperopt/curr_npy.npy"
    a=np.load(numpy_dir)
    sorted=np.sort(a)
    n=3
    res=  sorted[-n :] 
    print(f"rrr res {res} sorted {sorted}")   

    return np.mean(res)
# storage="mysql://root@34.90.134.17/testt"
study = optuna.create_study(
        study_name=experiment_name
        ,sampler=optuna.samplers.CmaEsSampler()    
        # ,sampler=optuna.samplers.NSGAIISampler()    
        ,pruner=optuna.pruners.HyperbandPruner()
        # ,storage=f"mysql://root:jm@34.90.134.17:3306/{experiment_name}"
        ,storage=f"mysql://root@34.90.134.17/{experiment_name}"
        ,load_if_exists=True
        ,direction="maximize"
        )

  
#         #mysql://root@localhost/example
study.optimize(objective, n_trials=400)

# my_proj_name='hyperparam_classic_aug' tag='hyperparam_classic_aug' my_proj_desc='debug' nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl