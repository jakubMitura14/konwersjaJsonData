import optuna
import os
from subprocess import Popen

os.environ['attn_num_mem_kv'] = '0'
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
os.environ['residual_attn'] = '0'
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


cmd=f"my_proj_name='seg lesions debug' tag='priming test' my_proj_desc='l4a test' nnUNetv2_train 101 3d_lowres 0 -tr Main_trainer_pl"

# p = Popen(cmd, shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
p = Popen(cmd, shell=True)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE

p.wait()

