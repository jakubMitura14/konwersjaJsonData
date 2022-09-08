FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

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

RUN apt-get update && apt-get install -y --no-install-recommends \
   curl \
   build-essential \
   coreutils \
   gcc \
   gettext \
   zlib1g-dev \
   openssl \
   perl \
   wget \
   zlib1g


## installing github CLI - https://github.com/cli/cli/blob/trunk/docs/install_linux.md
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
RUN sudo apt update
RUN sudo apt install gh
RUN apt autoremove python3 -y



RUN mkdir /app
WORKDIR /app

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda-11.3
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
## { Mitura end}
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates gnupg2 less sudo \
 # Spicy build and test dependencies.
 && apt-get install -y --no-install-recommends git cmake ninja-build ccache bison flex g++ libfl-dev zlib1g-dev libssl-dev jq locales-all make \
 # Spicy doc dependencies.
 && apt-get install -y --no-install-recommends python3 python3-pip python3-sphinx python3-sphinx-rtd-theme python3-setuptools python3-wheel doxygen \
 && pip3 install "btest>=0.66" pre-commit \
 # Install a recent CMake.
 && mkdir -p /opt/cmake \
 && curl -L https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0-Linux-x86_64.tar.gz | tar xzvf - -C /opt/cmake --strip-components 1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/cmake/bin:${PATH}"


RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR ${HOME}
USER ${NB_USER}

RUN pip3 --no-cache-dir install \
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
    pandas \
    comet_ml \
    jax[cpu] \
    optax \
    ott-jax \
    pydicom-seg


RUN export GITHUB_USER=jakubMitura14

ENV env_var_name=$argTokenGithub

RUN export GITHUB_TOKEN=argToken

RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l



RUN mkdir ${HOME}/data
RUN mkdir ${HOME}/data/scripts

RUN mkdir ${HOME}/externalRepos
RUN mkdir ${HOME}/forLesionAnalysis

RUN mkdir ${HOME}/externalRepos/elastix
RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix


RUN git clone https://github.com/jakubMitura14/konwersjaJsonData.git ${HOME}/data/scripts


# RUN cd ${HOME}/externalRepos/elastix && \
#     mkdir buildd && \
#     cd buildd && \
#     cmake ${HOME}/externalRepos/elastix/SuperBuild && \
#     make -j6 && \
#     cd ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python && \
#     python Packaging/setup.py install



# For backward compatibility we need this line. After 1.13 we can safely remove
# it.
ENV TF_NCCL_VERSION=

# Set TMP for nvidia build environment
ENV TMP="/tmp"

#RUN pip install itk
RUN pip install mdai
RUN pip install numpy
RUN pip install pandas
RUN pip install opencv-python
RUN pip install pydicom

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install more-itertools
RUN pip install SimpleITK
RUN pip install toolz
# RUN pip install SimpleITK-SimpleElastix==2.0.0rc2.dev910
# RUN mkdir ${HOME}/elastixx
# RUN git clone https://github.com/SuperElastix/elastix.git ${HOME}/elastixx


### DCMQI from https://hub.docker.com/r/qiicr/dcmqi/dockerfile




# RUN cd ${HOME}/elastixx && \
#     mkdir buildd && \
#     #cd buildd && \
#     ccmake -DITK USE 64BITS IDS =ON -DITK LEGACY REMOVE=ON -DCMAKE BUILD TYPE=“Release” -DCMAKE INSTALL PREFIX=${HOME}/elastixx/buildd  .; make
#     # cmake ${HOME}/elastixx && \
#     # make -j6

CMD ["/bin/bash"]

# RUN wget http://people.centos.org/tru/devtools-2/devtools-2.repo
# ENV CC /opt/rh/devtoolset-2/root/usr/bin/gcc
# ENV CXX /opt/rh/devtoolset-2/root/usr/bin/g++
# ENV FC /opt/rh/devtoolset-2/root/usr/bin/gfortran

# RUN mkdir ${HOME}/externalRepos/qiicr
# RUN git clone https://github.com/QIICR/dcmqi.git ${HOME}/externalRepos/qiicr

# # RUN export PATH=$PATH:/usr/lib/gcc
# # RUN export CC=gcc
# RUN cd ${HOME}/externalRepos/qiicr && \
#     mkdir buildd && \
#     cd buildd && \
#     cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc ${HOME}/externalRepos/qiicr && \
#     make -j6