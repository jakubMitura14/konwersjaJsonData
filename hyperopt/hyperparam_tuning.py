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
import json
import shutil
import glob


def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

# csv_dir="/workspaces/konwersjaJsonData/hyperopt/curr_csv.csv"
# curr_csv = pd.DataFrame([{"ress":0.0}])

# print(f"rrrrrrrrr {curr_csv}")
# curr_csv.to_csv(csv_dir) 

json_pathh='/workspaces/konwersjaJsonData/hyperopt/curr_json.json'
results_folder="/home/sliceruser/nnUNet_results/Dataset101_Prostate/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_0"
# os.makedirs(results_folder,exist_ok=True)



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

    alpha_low=trial.suggest_float("alpha_low", 0.0, 1000.0)
    os.environ['alpha_low'] = str(alpha_low)
    os.environ['alpha_high'] = str(alpha_low+trial.suggest_float("alpha_high", 0.0, 2000.0))

    # sigma_low=trial.suggest_float("sigma_low", 0.0, 19.0)
    # os.environ['sigma_low'] = str(sigma_low)
    # os.environ['sigma_high'] = str(sigma_low+trial.suggest_float("sigma_high", 0.0, 20.0))

    os.environ['GaussianBlurTransform'] = str(trial.suggest_float("GaussianBlurTransform", 0.0,0.9))

    os.environ['SimulateLowResolutionTransform'] = str(trial.suggest_float("SimulateLowResolutionTransform", 0.0,0.9))
    os.environ['GammaTransform_a'] = str(trial.suggest_float("GammaTransform_a", 0.0,0.9))
    os.environ['GammaTransform_b'] = str(trial.suggest_float("GammaTransform_b", 0.0,0.9))
    os.environ['p_scale_per_sample'] = str(trial.suggest_float("p_scale_per_sample", 0.0,0.9))

    scale_low=trial.suggest_float("scale_low", 0.0,1.0)
    os.environ['scale_low'] = str(scale_low)
    os.environ['scale_high'] = str(scale_low+trial.suggest_float("scale_high", 0.0,3.0))



    # os.environ['p_rot_per_sample'] = str(trial.suggest_float("p_rot_per_sample", 0.0,0.9))
    baseLr=0.00831
    os.environ['learning_rate'] = str(baseLr)#str(trial.suggest_float("learning_rate", baseLr/50,baseLr*2))



    ###os.environ['attn_num_mem_kv'] = '0'
    # os.environ['alpha_low'] = str(421.60033013863716)
    # os.environ['alpha_high'] = str(421.60033013863716+1451.4488917182302)

    os.environ['sigma_low'] = str(11.541639008878207)
    os.environ['sigma_high'] = str(11.541639008878207+ 9.669272682121036)

    os.environ['p_rot_per_axis']=trial.suggest_categorical("p_rot_per_axis", ["1", "2","3"])
    os.environ['independent_scale_for_each_axis']=trial.suggest_categorical("independent_scale_for_each_axis", ["1","0" ])
    
    # gmma_bottom=trial.suggest_float("gamma_bottom_c", 0.0,3.0)
    # os.environ['gamma_bottom_c']=str(gmma_bottom)
    # os.environ['gamma_up_c']=str(trial.suggest_float("gamma_up_c",  0.0,3.0)+gmma_bottom)
    # os.environ['gamma_prob_c']=str(trial.suggest_float("gamma_prob_c", 0.0,1.0))

    os.environ['RicianNoiseTransform'] = str(0.40230497711508434)
    # os.environ['GaussianBlurTransform'] = str(0.30684212800169935)

    os.environ['ContrastAugmentationTransform'] = str(0.6110392464376544)
    # os.environ['SimulateLowResolutionTransform'] = str(0.51587313094672182)
    # os.environ['GammaTransform_a'] = str(0.09474295989278318)
    # os.environ['GammaTransform_b'] = str(0.3249770010008122)
    os.environ['p_el_per_sample'] = str(0.3338220211800589)
    # os.environ['p_scale_per_sample'] = str(0.6832566131656115)
    os.environ['p_rot_per_sample'] = str(0.5495580611394884)
    # baseLr=0.00831
    # os.environ['learning_rate'] = str(0.03142343750219331)



# alpha_low 421.60033013863716

# alpha_high 1451.4488917182302

# sigma_low 11.541639008878207

# sigma_high 9.669272682121036

# RicianNoiseTransform 0.40230497711508434

# GaussianBlurTransform 0.30684212800169935

# ContrastAugmentationTransform 0.6110392464376544

# SimulateLowResolutionTransform 0.5158731309467218

# GammaTransform_a 0.09474295989278318

# GammaTransform_b 0.3249770010008122

# p_el_per_sample 0.3338220211800589

# p_scale_per_sample 0.3757931967159725

# p_rot_per_sample 0.6832566131656115

# learning_rate 0.03142343750219331

def set_norm_and_bias_field(trial):
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


# experiment_name="general_augment"
# experiment_name="classic_augmentations2"#bias_norm
# experiment_name="test"#bias_norm
experiment_name="classic_augmentations9"#bias_norm


def setup_pseudo_lesion_adder_and_loss(trial):
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
    #for custom loss function
    # os.environ['w0'] = str(trial.suggest_float("w0", 0.0,1.0))
    # os.environ['w1'] = str(trial.suggest_float("w1", 0.0,1.0))
    # os.environ['w2'] = str(trial.suggest_float("w2", 0.001,0.999))



# trial=[]
# set_norm_and_bias_field(trial)
# seg_lesions_custom.main_func()

def objective(trial: optuna.trial.Trial) -> float:


        
    #checking if there is some failed trial if so we will restart it
    # expId = RetryFailedTrialCallback.retried_trial_number(trial)
    # if(expId is None):
    expId=trial._trial_id
    print(f"cccccc current {expId}")
    save_trial_id(expId)

    set_norm_and_bias_field(trial)
    # seg_lesions_custom.main_func()
    set_env_variables_for_general_transforms(trial)
    setup_pseudo_lesion_adder_and_loss(trial)

    # cmd=f"custom_aug_loss tag='custom_aug_loss' my_proj_desc='custom_aug_loss' nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl"
    cmd=f" my_proj_name='classic_augmentations2' tag='classic_augmentations2' my_proj_desc='classic_augmentations2' nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl"

    # p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
    p = Popen(cmd, shell=True)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE

    p.wait()
    os.environ['load_checkpoint'] = "0"
    shutil.rmtree(results_folder)
    os.mkdir(results_folder)    
    os. remove('/workspaces/konwersjaJsonData/hyperopt/curr_npy.npy') 

    with open('/workspaces/konwersjaJsonData/hyperopt/curr_npy.npy', 'wb') as f:
        np.save(f, np.array([0]))
    
    print("ssssave trial id as empty")
    save_trial_id(" ")# reset trial id


    numpy_dir="/workspaces/konwersjaJsonData/hyperopt/curr_npy.npy"
    a=np.load(numpy_dir)
    res=  np.max((np.roll(a,1)+a+np.roll(a,-1))/3)
    print(f"rrrr res {res} aa {a}")   
    
    
    return np.max((np.roll(a,1)+a+np.roll(a,-1))/3)
    # return np.max(a)
# storage="mysql://root@34.90.134.17/testt"

storage = optuna.storages.RDBStorage(
    url=f"mysql://root@34.90.134.17/{experiment_name}",
    # engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)

def get_trial_id():
    try:
        with open(json_pathh, 'r') as openfile:
        
            # Reading from json file
            json_object = json.load(openfile)
            return json_object["id"]
    except:
        return " "

def save_trial_id(trial_id):
    dictionary = {
        "id": str(trial_id)
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    
    # Writing to sample.json
    with open(json_pathh, "w") as outfile:
        outfile.write(json_object)    
    return trial_id

study = optuna.create_study(
        study_name=experiment_name
        # ,sampler=optuna.samplers.CmaEsSampler()    
        ,sampler=optuna.samplers.NSGAIISampler()    
        # ,pruner=optuna.pruners.HyperbandPruner()
        # ,storage=f"mysql://root:jm@34.90.134.17:3306/{experiment_name}"
        ,storage=f"mysql://root@34.90.134.17/{experiment_name}"
        ,load_if_exists=True
        ,direction="maximize"
        )

#         #mysql://root@localhost/example
old_trial_id=get_trial_id()
print(f"ooooooooooooooooooooooo {old_trial_id}")
if(old_trial_id==" "):
    os.environ['load_checkpoint'] = "0"
    os.makedirs(results_folder,exist_ok=True)
    shutil.rmtree(results_folder)
    os.mkdir(results_folder)    
    os. remove('/workspaces/konwersjaJsonData/hyperopt/curr_npy.npy') 
    with open('/workspaces/konwersjaJsonData/hyperopt/curr_npy.npy', 'wb') as f:
        np.save(f, np.array([0]))

    
    
    study.optimize(objective, n_trials=900,gc_after_trial=True)
else:    
    print(f"starting from old trial id {old_trial_id}")    
    os.environ['load_checkpoint'] = "1"
    path = results_folder+r'/*.ckpt'
    files = glob.glob(path)
    print(files)
    
    
    os.environ['checkPoint_path'] = files[-1]
        

    
    
    
    # frozen=storage._build_frozen_trial_from_trial_model(storage.get_trial(int(old_trial_id)))
    # frozen=storage.get_trial(int(old_trial_id))
    # print(f"aaa {type(frozen)}")
    # study.add_trial(frozen)
    # distribs=storage.get_trial(int(old_trial_id)).distributions
    # # study.add_trial(storage.get_trial(int(old_trial_id)))
    failed_trial_number = int(old_trial_id)
    failed_trial = storage.get_trial(int(old_trial_id))#study.get_trials()[failed_trial_number]
    study.enqueue_trial(failed_trial.params)
    study.optimize(objective, n_trials=900,gc_after_trial=True)

    
    
    
    
    
    # study.tell(asked)
#     # study.optimize(objective, n_trials=900,gc_after_trial=True)
# storage.get_trial_params
# storage._build_frozen_trial_from_trial_model(storage.get_trial(int(old_trial_id)))
# study.add_trial()
#last was 1211

# optuna-dashboard mysql://root@34.90.134.17/classic_augmentations7
# optuna-dashboard mysql://root@34.90.134.17/custom_aug_loss_f
# # my_proj_name='hyperparam_classic_aug' tag='hyperparam_classic_aug' my_proj_desc='debug' nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl