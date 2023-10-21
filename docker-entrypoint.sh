#!/usr/bin/env bash

# echo "Hello from our entrypoint!"
#at entrypoint as CUDA context is not available during build ... 

# git clone https://github.com/microsoft/Swin3D.git ${HOME}/Swin3d \
# && cd ${HOME}/Swin3d \
# && python3 -m pip install -r requirements.txt \
# && python3 setup.py install

# git clone https://github.com/zsef123/Connected_components_PyTorch.git ${HOME}/connected_comp \
# && cd ${HOME}/connected_comp \
# && python3 setup.py install

cd /workspaces/konwersjaJsonData/nnunet/nnunetv2pl/nnUNet \
&& python3 -m pip  install -e .


cd /workspaces/konwersjaJsonData \
&&  python3 -m infrence.prprocess

# python3 /workspaces/konwersjaJsonData/hyperopt/hyperparam_tuning.py


# set -ex

# script_dir=$(cd $(dirname $0) || exit 1; pwd)

# ################################################################################
# # Set up headless environment
# source $script_dir/start-xorg.sh



export SHELL=/bin/bash



exec "$@"

