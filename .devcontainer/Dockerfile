# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
################################################################################
# Prevent apt-get from prompting for keyboard choice
#  https://superuser.com/questions/1356914/how-to-install-xserver-xorg-in-unattended-mode
ENV DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    wget\
    manpages-dev\
    g++\
    gcc\
    nodejs\
    libssl-dev\
    unzip\
    libidn11-dev\
    libglu1-mesa-dev \
    libpangox-1.0-dev \
    libpangoxft-1.0-0 \
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
    && rm -rf /var/lib/apt/lists/*


# ENV PATH="/opt/cmake/bin:${PATH}"
ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
#     python3.9 get-pip.py && \
#     rm get-pip.py

# RUN python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip  --no-cache-dir install \
    pandas==1.5.3 \
    future>=0.17.1 \
    grpcio \
    h5py \
    mock \
    numpy \
    portpicker \
    requests \
    kymatio==0.2.1 \
    h5py==3.7.0 \
    #itk \
    #SimpleITK>=2.1.1.2 \
    # pandas \
    comet_ml \
    jax[cpu] \
    optax \
    ott-jax \
    pydicom-seg \
    numpy \
    opencv-python \
    pydicom \
    more-itertools \
    SimpleITK \
    highdicom==0.19.0 \
    # nnunet==1.7.0 \
    multiprocess \
    dill \
    xnat \
    pyxnat\
    mdai \
    pyradiomics \ 
    six \
    napari-simpleitk-image-processing \
    pytorch-lightning==2.0.7 \
    itk itk-elastix monai-weekly \
    pymia einops seaborn \
    optuna

RUN python3 -m pip  --no-cache-dir install torchio

RUN export GITHUB_USER=jakubMitura14

ENV env_var_name=$argTokenGithub

RUN export GITHUB_TOKEN=argToken

RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l



RUN mkdir ${HOME}/scripts
RUN mkdir ${HOME}/externalRepos
RUN mkdir ${HOME}/forLesionAnalysis



# Set TMP for nvidia build environment
ENV TMP="/tmp"


ENV nnUNet_raw="/home/sliceruser/nnunetMainFolder/nnUNet_raw"
ENV nnUNet_preprocessed="/home/sliceruser/nnunetMainFolder/nnUNet_preprocessed"
ENV nnUNet_results="/home/sliceruser/nnUNet_results"

CMD ["/bin/bash"]

# cd /workspaces/konwersjaJsonData/nnunet/nnunetv2pl/nnUNet
# python3 -m pip  install -e .
