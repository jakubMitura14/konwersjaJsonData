# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

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
    # libpangox-1.0-dev \
    # libpangoxft-1.0-0 \
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

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 && \
    apt-get install -y  python3.10-dev



RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      cmake \
      g++ \
      gfortran \
      git \
      vim \
    #   hdf5-tools \
      libmkl-full-dev \ 
      libboost-dev \
      libfftw3-dev \
      libnfft3-dev \
      libgfortran5 \
      libgmp-dev \
    #   libhdf5-dev \
      libopenmpi-dev \
      cython3 \
      openmpi-bin \
      openmpi-common \
      openmpi-doc \
      python3-dev \
      python3-mako \
      python3-numpy \
      python3-scipy \
      python3-matplotlib \
      python3-mpi4py \
      python3-pip \
      python3-sphinx \
      python3-skimage \ 
      python3-nbsphinx \
      python3-notebook \
      libpython3-dev \
      git \
      ca-certificates \
      openssl \
      rsync \
      libomp-dev \
      nano \
      less \
      # Jupyter lab packages
      nodejs \
    #   texlive \
    #   dvipng \
    #   texlive-latex-extra \
    #   texlive-fonts-recommended \
      && \
      apt-get autoremove --purge -y && \
      apt-get autoclean -y && \
      rm -rf /var/cache/apt/* /var/lib/apt/lists/*


RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# RUN python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip  --no-cache-dir install \
    pandas==1.5.3 \
    future>=0.17.1 \
    grpcio \
    mock \
    numpy \
    portpicker \
    requests \
    kymatio==0.2.1 \
    # h5py\
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
    optuna \
    torchio \
    'itkwidgets[all]>=1.0a23' \
    connected-components-3d \
    ninja


     
RUN python3 -m pip  --no-cache-dir install install --no-cache-dir \
    cython \
    decorator \
    jupyter \
    ipywidgets \
    pytest \
    pandas \
    shapely \
    monty \
    mpi4py \
    pymatgen \
    ase \ 
    Jinja2 \
    Mako \
    tornado \
    zmq \
    notebook \
    tk \
    myst_parser \ 
    sphinx_rtd_theme  \
    linkify-it-py \ 
    Pandoc \
    # Jupyter lab packages
    jupyterlab \
    descartes \
    ipympl \
    RISE 


# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.1-535.86.10-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.1-535.86.10-1_amd64.deb
# RUN cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
# RUN apt-get update
# RUN apt-get -y install cuda

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb  \
# && dpkg -i cuda-keyring_1.1-1_all.deb  \
# && apt-get update \
# && apt-get -y install cuda




# ENV HOME /home/$USER
# ENV MPI_DIR=/opt/ompi
# ENV PATH="$MPI_DIR/bin:$HOME/.local/bin:$PATH"
# ENV LD_LIBRARY_PATH="$MPI_DIR/lib:$LD_LIBRARY_PATH"
# WORKDIR $HOME







RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda


# RUN wget -q -O - http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz | tar -C /tmp -xzf - && \
#     cd /tmp/mpich-3.3.2 && \
#     ./configure && \
#     make -j 4 && \
#     make install && \
#     rm -rf /tmp/mpich-3.3.2




#--enable-fortran
RUN mkdir ${HOME}/hdf5
RUN mkdir ${HOME}/hdf5/dest
RUN mkdir ${HOME}/hdf5/inst

RUN python3 -m pip uninstall -y h5py

RUN wget -q -O - https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz | tar -C /tmp -xzf - && \
    cd /tmp/hdf5-1.12.0 && \
    ./configure --prefix=${HOME}/hdf5/inst --enable-parallel --enable-shared && \
    make DESTDIR=${HOME}/hdf5/dest && \
    make -j 4 && \
    make install && \
    rm -rf /tmp/hdf5-1.12.0

RUN CC=mpicc HDF5_MPI="ON" HDF5_VERSION=1.12.0 HDF5_DIR=${HOME}/hdf5/inst python3 -m pip install --no-binary=h5py h5py
RUN export GITHUB_USER=jakubMitura14
RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l


RUN mkdir ${HOME}/scripts \
    && mkdir ${HOME}/externalRepos \
    && mkdir ${HOME}/forLesionAnalysis \
    && mkdir ${HOME}/Swin3d \
    && mkdir ${HOME}/connected_comp
# RUN mkdir ${HOME}/connected_comp


# CUDA_HOME=/usr/local/cuda-X.X


# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-465.19.01-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
# RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
# RUN apt-get update
# RUN apt-get -y install cuda-11.3
# RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub


# RUN git clone https://github.com/zsef123/Connected_components_PyTorch.git ${HOME}/connected_comp \
# && cd ${HOME}/connected_comp \
# && python3 setup.py install

# RUN python3 -m pip install ninja timm==0.4.9 -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
# RUN git clone https://github.com/microsoft/Swin3D.git ${HOME}/Swin3d \
# && cd ${HOME}/Swin3d \
# && python3 -m pip install -r requirements.txt \
# && python3 setup.py install
RUN python3 -m pip install transformers datasets lightning==2.0.7


# Set TMP for nvidia build environment
ENV TMP="/tmp"


ENV nnUNet_raw="/home/sliceruser/nnunetMainFolder/nnUNet_raw"
ENV nnUNet_preprocessed="/home/sliceruser/nnunetMainFolder/nnUNet_preprocessed"
ENV nnUNet_results="/home/sliceruser/nnUNet_results"

# CMD ["/bin/bash"]
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT [ "/docker-entrypoint.sh" ]
CMD [ "sleep", "infinity" ]

# cd /workspaces/konwersjaJsonData/nnunet/nnunetv2pl/nnUNet
# python3 -m pip  install -e .
