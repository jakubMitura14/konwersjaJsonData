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
    vim net-tools curl \
    libgl1-mesa-glx \
    xserver-xorg-video-dummy \
    libxrender1 \
    libpulse0 \
    libpulse-mainloop-glib0  \
    libnss3  \
    libxcomposite1 \
    libxcursor1 \
    libfontconfig1 \
    libxrandr2 \
    libasound2 \
    libglu1 \
    x11vnc \
    awesome \
    jq \
    git \
    nautilus \
    jupyter-core \
    unzip \
    default-libmysqlclient-dev

#   apt-get autoremove --purge -y && \
#   apt-get autoclean -y && \
#   rm -rf /var/cache/apt/* /var/lib/apt/lists/*


# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp


RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py


#--enable-fortran
RUN mkdir ${HOME}/hdf5  \
    && mkdir ${HOME}/hdf5/dest  \
    && mkdir ${HOME}/hdf5/inst \
    && mkdir ${HOME}/Slicer \
    && mkdir ${HOME}/Slicer/bin \
    && mkdir ${HOME}/Slicer/bin/websockify 


COPY websockify ./Slicer/bin/
RUN chmod +x ${HOME}/Slicer/bin/websockify


RUN wget -q -O - https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz | tar -C /tmp -xzf - && \
    cd /tmp/hdf5-1.12.0 && \
    ./configure --prefix=${HOME}/hdf5/inst --enable-parallel --enable-shared && \
    make DESTDIR=${HOME}/hdf5/dest && \
    make -j 4 && \
    make install && \
    rm -rf /tmp/hdf5-1.12.0



RUN mkdir ${HOME}/scripts \
    && mkdir ${HOME}/externalRepos \
    && mkdir ${HOME}/forLesionAnalysis \
    && mkdir ${HOME}/Swin3d \
    && mkdir ${HOME}/connected_comp \
    && mkdir ${HOME}/NBIA

# RUN mkdir ${HOME}/connected_comp

## for downloads from cancer imaging archive
RUN apt-get install -y ruby-dev  \
    && apt install -y libtool libffi-dev ruby ruby-dev make  \
    && apt install -y libzmq3-dev libczmq-dev  \
    && gem install ffi-rzmq rake  \
    && gem install specific_install  \
    && gem specific_install https://github.com/SciRuby/iruby  \
    && iruby register  \
    && gem install net-protocol -v 0.1.2  \
    && gem install net-smtp -v 0.3.0  \
    && gem install aspera-cli 
RUN ascli conf ascp install


# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN wget https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
    tar xzvf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0 && \
    rm -rf ./boost_1_68_0.tar.gz 



RUN mkdir /usr/share/desktop-directories/ \
    && wget -P ~/NBIA-Data-Retriever https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4.1/nbia-data-retriever-4.4.1.deb \
    && sudo apt-get -y install xdg-utils \
    && dpkg-deb -x ~/NBIA-Data-Retriever/nbia-data-retriever-4.4.1.deb ${HOME}/NBIA


RUN python3 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 
# RUN python3 -m pip install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2--index-url https://download.pytorch.org/whl/cu121


RUN python3 -m pip  --no-cache-dir install install --no-cache-dir \
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
    # pytorch-lightning==2.0.7 \
    # itk itk-elastix monai-weekly \
    pymia einops seaborn \
    optuna \
    torchio \
    connected-components-3d \
    ninja \
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
    RISE \
    transformers datasets deepspeed==0.10.1 ctviewer more_itertools==10.1.0 tcia_utils==1.6.9 \
    rt-utils \
    scikit-image \
    itk==5.3.0 \
    vtk \
    monai[nibabel,itk,tqdm]  \
    nibabel \
    tqdm \
    itkwidgets[all]==1.0a23  \
    imjoy-elfinder \
    imjoy-jupyter-extension \
    imjoy-jupyterlab-extension \
    llvmlite \
    numba \
    tensorboardX \
    easydict \
    pyyaml \
    scikit-image \
    tqdm \
    SharedArray \
    six \
    evalutils==0.3.0 \
    https://github.com/DIAGNijmegen/picai_prep/archive/refs/tags/v1.3.2.zip \
    pytorch-ignite==0.4.12 \
    xformers==0.0.21 triton fairscale==0.4.13 hiddenlayer==0.3 x-transformers==1.19.1 intensity-normalization==2.2.4  \
    lightning==2.1.0rc0 \
    spconv-cu102 mysqlclient lapgm[gpu]

RUN python3 -m pip uninstall -y h5py


RUN CC=mpicc HDF5_MPI="ON" HDF5_VERSION=1.12.0 HDF5_DIR=${HOME}/hdf5/inst python3 -m pip install --no-binary=h5py h5py \
    && export GITHUB_USER=jakubMitura14
RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l




# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
#     && mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
#     && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
#     && dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
#     && cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/  \
#     && apt-get update  \
#     && apt-get -y install cuda


# RUN wget -q -O - http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz | tar -C /tmp -xzf - && \
#     cd /tmp/mpich-3.3.2 && \
#     ./configure && \
#     make -j 4 && \
#     make install && \
#     rm -rf /tmp/mpich-3.3.2





# /home/sliceruser/NBIA/opt/nbia-data-retriever/nbia-data-retriever --cli /workspaces/konwersjaJsonData/nnunet/pretraining/manifests/UPENN-GBM_DownloadManifest20221129.tcia -d /home/sliceruser/preprocessed/raw -l /home/sliceruser/NBIA/passwd.txt -v –f

# RUN python3 -m pip install xformers==0.0.21 triton fairscale==0.4.13 hiddenlayer==0.3 x-transformers==1.19.1 intensity-normalization==2.2.4 lightning==2.1.0rc0
# # RUN mkdir ${HOME}/bolts \
# #     && cd ${HOME}/bolts \
# #     && git clone https://github.com/Lightning-Universe/lightning-bolts.git ${HOME}/bolts \
# #     && python3 -m pip  install -e .
# RUN python3 -m pip install install lightning==2.1.0rc0

# RUN mkdir ${HOME}/lighting \
#     && cd ${HOME}/lighting \
#     && git clone https://github.com/Lightning-AI/lightning.git ${HOME}/lighting \
#     && python3 -m pip  install -e .



RUN mkdir ${HOME}/x_transformer \
    && cd ${HOME}/x_transformer \
    && git clone https://github.com/lucidrains/x-transformers.git ${HOME}/x_transformer \
    && python3 -m pip  install -e .



# Install Boost geometry


RUN pip uninstall -y SharedArray && \
    pip install SharedArray 





# python3 -m pip install easydict termcolor tensorboardX pybind11 portalocker lark fire ccimport pccm cumm spconv pcdet
# python3 -m pip install pcdet
# RUN mkdir ${HOME}/VOTR \
# && cd ${HOME}/VOTR \
# && git clone https://github.com/PointsCoder/VOTR.git ${HOME}/VOTR \
# && python3 -m pip  install -e .



# cd /workspaces/konwersjaJsonData/nnunet/nnunetv2pl/nnUNet
# python3 -m pip  install -e .

RUN mkdir ${HOME}/apex \
    && git clone https://github.com/NVIDIA/apex ${HOME}/apex \
    && cd ${HOME}/apex \
    && python3 -m pip  install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# Set TMP for nvidia build environment
ENV TMP="/tmp"

# RUN pip install spconv-cu102 mysqlclient lapgm[gpu]


ENV nnUNet_raw="/home/sliceruser/nnunetMainFolder/nnUNet_raw"
ENV nnUNet_preprocessed="/home/sliceruser/nnunetMainFolder/nnUNet_preprocessed"
ENV nnUNet_results="/home/sliceruser/nnUNet_results"
ENV COMET_API_KEY="yB0irIjdk9t7gbpTlSUPnXBd4"
# CMD ["/bin/bash"]
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT [ "/docker-entrypoint.sh" ]
# CMD ["sleep", "infinity" ,"sh", "-c", "./Slicer/bin/PythonSlicer -m jupyter notebook --port=$JUPYTERPORT --ip=0.0.0.0 --no-browser --NotebookApp.default_url=/lab/"]
CMD ["sleep", "infinity"]
################################################################################
# Install Slicer application startup script

# COPY .slicerrc.py .


################################################################################
# Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG IMAGE
ARG VCS_REF
ARG VCS_URL

# cd /workspaces/konwersjaJsonData/nnunet/nnunetv2pl/nnUNet
# python3 -m pip  install -e .

# cp /home/sliceruser/nnunetMainFolder/nnUNet_raw/Dataset294_Prostate/labelsTr/9002700.nii.gz /workspaces/konwersjaJsonData/explore/for_explore/9002700.nii.gz