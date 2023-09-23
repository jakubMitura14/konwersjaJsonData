import optuna
import os
from subprocess import Popen
import subprocess
from optuna.storages import RetryFailedTrialCallback
import importlib.util
import importlib
import sys

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

seg_lesions_custom=loadLib("seg_lesions_custom","/workspaces/konwersjaJsonData/nnunet/seg_lesions_custom.py")

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
    os.environ['alpha_low'] = alpha_low
    os.environ['alpha_high'] = alpha_low+trial.suggest_float("alpha_low", 0.0, 2000.0)

    sigma_low=trial.suggest_float("sigma_low", 0.0, 19.0)
    os.environ['sigma_low'] = sigma_low
    os.environ['sigma_high'] = sigma_low+trial.suggest_float("sigma_high", 0.0, 20.0)

    os.environ['p_rot_per_axis']=trial.suggest_categorical("p_rot_per_axis", ["1", "2","3"])

    os.environ['RicianNoiseTransform'] = trial.suggest_float("RicianNoiseTransform", 0.0,0.9)
    os.environ['RicianNoiseTransform'] = trial.suggest_float("RicianNoiseTransform", 0.0,0.9)



def set_norm_and_bias_field(trial):
    os.environ['to_include_normalize'] = trial.suggest_categorical("to_include_normalize", ["t2w_adc_hbv", "t2w_adc","t2w_hbv","t2w"])
    os.environ['tau'] = f"{trial.suggest_float('tau',5e-7,5e-2)}"
    os.environ['n_classes'] = f"{trial.suggest_int( 'n_classes', 1,20)}"
    os.environ['log_initialize'] = trial.suggest_categorical("log_initialize", ["0", "1"])

set_env_variables_for_swin()
os.environ['best_metric'] ='0.0'


# experiment_name="general_augment"
experiment_name="bias_norm"#bias_norm


def objective(trial: optuna.trial.Trial) -> float:

    #checking if there is some failed trial if so we will restart it
    expId = RetryFailedTrialCallback.retried_trial_number(trial)
    if(expId is None):
        expId=trial.number

    set_norm_and_bias_field(trial)
    seg_lesions_custom.main_func()

    cmd=f"my_proj_name='seg lesions debug' tag='priming test' my_proj_desc='l4a test' nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl"

    # p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
    p = Popen(cmd, shell=True)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE

    p.wait()


    return float(os.getenv('best_metric'))

# storage="mysql://root@34.90.134.17/testt"
study = optuna.create_study(
        study_name=experiment_name
        ,sampler=optuna.samplers.NSGAIISampler()    
        ,pruner=optuna.pruners.HyperbandPruner()
        # ,storage=f"mysql://root:jm@34.90.134.17:3306/{experiment_name}"
        ,storage=f"mysql://root@34.90.134.17/{experiment_name}"
        ,load_if_exists=True
        ,direction="maximize"
        )

  
#         #mysql://root@localhost/example
study.optimize(objective, n_trials=400)
