# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
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
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
    && rm -rf /var/lib/apt/lists/*



## installing github CLI - https://github.com/cli/cli/blob/trunk/docs/install_linux.md
# RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
# # RUN sudo apt update
# # RUN apt-key adv --fetch-keys 'https://packages.sury.org/php/apt.gpg' > /dev/null 2>&1
# # RUN sudo apt install gh

# # RUN apt-get clean
# # RUN mv /var/lib/apt/lists /tmp
# # RUN mkdir -p /var/lib/apt/lists/partial
# # RUN apt-get clean
# # RUN apt-get update
# # RUN gpg --keyserver keyserver.ubuntu.com --recv-keys 16126D3A3E5C1192 
# # RUN apt-key adv --fetch-keys 'https://packages.sury.org/php/apt.gpg' > /dev/null 2>&1
# RUN gpg --keyserver hkp://subkeys.pgp.net --recv-keys 16126D3A3E5C1192
# RUN gpg --export --armor 16126D3A3E5C1192 | sudo apt-key add - 
# RUN apt-get update

# RUN apt-key adv --refresh-keys --keyserver keyserver.ubuntu.com
# RUN rm -rf /var/lib/apt/lists
# RUN mkdir /var/lib/apt/lists

# RUN apt autoremove python3 -y

# RUN apt install software-properties-common && \
#     apt update && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt install python3.9

RUN apt-get update && apt-get install --no-install-recommends -y python3.9 python3.9-dev python3.9-venv python3-pip python3-wheel build-essential && \
	apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub



# RUN mkdir /app
# WORKDIR /app

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
# RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
# RUN apt-get update
# RUN apt-get -y install cuda-11.3
# RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
# ## { Mitura end}
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends curl ca-certificates gnupg2 less sudo \
#  # Spicy build and test dependencies.
#  && apt-get install -y --no-install-recommends git cmake ninja-build ccache bison flex g++ libfl-dev zlib1g-dev libssl-dev jq locales-all make \
#  # Spicy doc dependencies.
#  && apt-get install -y --no-install-recommends python3 python3-pip python3-sphinx python3-sphinx-rtd-theme python3-setuptools python3-wheel doxygen \
#  && pip3 install "btest>=0.66" pre-commit \
#  # Install a recent CMake.
#  && mkdir -p /opt/cmake \
#  && curl -L https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0-Linux-x86_64.tar.gz | tar xzvf - -C /opt/cmake --strip-components 1 \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*

# ENV PATH="/opt/cmake/bin:${PATH}"
ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}




RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

RUN python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3.9 -m pip  --no-cache-dir install \
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
    dill xnat pyxnat mdai




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

#RUN pip install itk

# RUN pip install SimpleITK-SimpleElastix==2.0.0rc2.dev910
# RUN mkdir ${HOME}/elastixx
# RUN git clone https://github.com/SuperElastix/elastix.git ${HOME}/elastixx



RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix

RUN chmod +x ${HOME}/externalRepos/elastix
#from https://installati.one/ubuntu/22.04/libpangoxft-1.0-0/
RUN apt update
RUN apt -y install libtbb-dev
#from https://askubuntu.com/questions/409310/error-while-loading-shared-libraries-cannot-open-shared-object-file-no-such-fi
RUN apt -y install libgtk-3-dev
RUN apt update

RUN apt-get -y install libidn11-dev
RUN apt-get -y install libglu1-mesa-dev
RUN apt-get -y install libpangox-1.0-dev
RUN apt-get -y install libpangoxft-1.0-0

RUN mkdir ${HOME}/elastixBase && \
cd ${HOME}/elastixBase && \
wget https://github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2&& \
tar -xf elastix-5.0.1-linux.tar.bz2


RUN mkdir ${HOME}/nnUNet  && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git  ${HOME}/nnUNet && \
    cd ${HOME}/nnUNet  && \
    python3.9 -m pip  install -e . 

ENV nnUNet_n_proc_DA=8
RUN python3.9 -m pip  --no-cache-dir install --upgrade git+https://github.com/julien-blanchon/hiddenlayer.git

RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so /usr/local/lib/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 /usr/local/lib/libANNlib-5.0.so.1

RUN ldconfig
RUN apt-get install -q -y vim nano 
RUN mkdir ${HOME}/registration
COPY registration/parameters.txt ${HOME}/registration/parameters.txt
COPY registration/parametersB.txt ${HOME}/registration/parametersB.txt

### DCMQI from https://hub.docker.com/r/qiicr/dcmqi/dockerfile


# sudo docker run --init  -e --gpus all --ipc host --privileged --net host -p 8888:8888 -p49053:49053  -v /mnt/disks/sdb:/home/sliceruser/data -it  ai4ar_prep:latest bash

# sudo docker run -it --init --gpus all --ipc host --privileged --net host -p 8888:8888 -p49053:49053  -v /mnt/disks/sdb:/home/sliceruser/data ai4ar:latest  bash


# RUN cd ${HOME}/elastixx && \
#     mkdir buildd && \
#     #cd buildd && \
#     ccmake -DITK USE 64BITS IDS =ON -DITK LEGACY REMOVE=ON -DCMAKE BUILD TYPE=“Release” -DCMAKE INSTALL PREFIX=${HOME}/elastixx/buildd  .; make
#     # cmake ${HOME}/elastixx && \
#     # make -j6

# ENV nnUNet_raw="/media/fabian/nnUNet_raw"
# ENV nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
# ENV nnUNet_results="/media/fabian/nnUNet_results"
ENV nnUNet_raw="/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw"
ENV nnUNet_preprocessed="/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_preprocessed"
ENV nnUNet_results="/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_results"

CMD ["/bin/bash"]


