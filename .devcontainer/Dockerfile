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
    git 
#   apt-get autoremove --purge -y && \
#   apt-get autoclean -y && \
#   rm -rf /var/cache/apt/* /var/lib/apt/lists/*



RUN apt-get install -y nautilus jupyter-core unzip


RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# RUN python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118




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
    pytorch-lightning==2.0.7 \
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
    transformers datasets deepspeed==0.10.1 ctviewer lightning==2.0.7 more_itertools==10.1.0 tcia_utils==1.6.9 \
    rt-utils \
    pydicom-seg  \
    scikit-image \
    itk==5.3.0 \
    vtk \
    monai[nibabel,itk,tqdm]  \
    nibabel \
    tqdm \
    itkwidgets[all]==1.0a23  \
    imjoy-elfinder \
    imjoy-jupyter-extension \
    imjoy-jupyterlab-extension





RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    && mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/  \
    && apt-get update  \
    && apt-get -y install cuda


# RUN wget -q -O - http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz | tar -C /tmp -xzf - && \
#     cd /tmp/mpich-3.3.2 && \
#     ./configure && \
#     make -j 4 && \
#     make install && \
#     rm -rf /tmp/mpich-3.3.2




#--enable-fortran
RUN mkdir ${HOME}/hdf5  \
    && mkdir ${HOME}/hdf5/dest  \
    && mkdir ${HOME}/hdf5/inst \
    && mkdir ${HOME}/Slicer \
    && mkdir ${HOME}/Slicer/bin \
    && mkdir ${HOME}/Slicer/bin/websockify 


COPY websockify ./Slicer/bin/
RUN chmod +x ${HOME}/Slicer/bin/websockify

RUN python3 -m pip uninstall -y h5py

RUN wget -q -O - https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz | tar -C /tmp -xzf - && \
    cd /tmp/hdf5-1.12.0 && \
    ./configure --prefix=${HOME}/hdf5/inst --enable-parallel --enable-shared && \
    make DESTDIR=${HOME}/hdf5/dest && \
    make -j 4 && \
    make install && \
    rm -rf /tmp/hdf5-1.12.0

RUN CC=mpicc HDF5_MPI="ON" HDF5_VERSION=1.12.0 HDF5_DIR=${HOME}/hdf5/inst python3 -m pip install --no-binary=h5py h5py \
    && export GITHUB_USER=jakubMitura14
RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l


RUN mkdir ${HOME}/scripts \
    && mkdir ${HOME}/externalRepos \
    && mkdir ${HOME}/forLesionAnalysis \
    && mkdir ${HOME}/Swin3d \
    && mkdir ${HOME}/connected_comp
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

RUN mkdir /usr/share/desktop-directories/  
# RUN cd /usr/share/desktop-directories/ 
RUN wget -P /content/NBIA-Data-Retriever https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4.1/nbia-data-retriever-4.4.1.deb   
# RUN dpkg -i /content/NBIA-Data-Retriever/nbia-data-retriever-4.4.1.deb



# COPY etc/dpkg/dpkg.cfg.d/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
# RUN echo 'APT::Install-Recommends "0" ; APT::Install-Suggests "0" ;' >> /etc/apt/apt.conf && \
#     echo 'Dir::Cache::pkgcache "";\nDir::Cache::srcpkgcache "";' | tee /etc/apt/apt.conf.d/00_disable-cache-files


# ARG SLICER_DOWNLOAD_URL=https://download.slicer.org/bitstream/6286ce04e8408647b39f803a

# RUN curl -JL "$SLICER_DOWNLOAD_URL" | tar xz -C /tmp && mv /tmp/Slicer* Slicer

# ENV VNCPORT=49053
# ENV JUPYTERPORT=8888
# ENV DISPLAY=:10

# COPY xorg.conf .

# ################################################################################
# # Set up remote desktop access - step 1/2

# # Build rebind.so (required by websockify)
# RUN set -x && \
#     apt-get update -q -y && \
#     apt-get install -y build-essential --no-install-recommends && \
#     mkdir src && \
#     cd src && \
#     git clone https://github.com/novnc/websockify websockify && \
#     cd websockify && \
#     make && \
#     cp rebind.so /usr/lib/ && \
#     cd .. && \
#     rm -rf websockify && \
#     cd .. && \
#     rmdir src && \
#     apt-get purge -y --auto-remove build-essential

# # Set up launcher for websockify
# # (websockify must run in  Slicer's Python environment)
# RUN chmod +x ${HOME}/Slicer/bin/websockify


# RUN mkdir ${HOME}/install
# COPY start-xorg.sh ${HOME}/install/start-xorg.sh
# COPY install.sh ${HOME}/install/install.sh
# RUN chmod +rwx -R ${HOME}/install
# RUN ${HOME}/install/install.sh ${HOME}/Slicer/Slicer 

# # Now install websockify and jupyter-server-proxy (fixed at tag v1.6.0)
# RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --upgrade websockify && \
#     cp /usr/lib/rebind.so /home/sliceruser/Slicer/lib/Python/lib/python3.9/site-packages/websockify/ && \
#     /home/sliceruser/Slicer/bin/PythonSlicer -m pip install notebook jupyterhub jupyterlab && \
#     /home/sliceruser/Slicer/bin/PythonSlicer -m pip install -e \
#         git+https://github.com/lassoan/jupyter-desktop-server#egg=jupyter-desktop-server \
#         git+https://github.com/jupyterhub/jupyter-server-proxy@v1.6.0#egg=jupyter-server-proxy



# ################################################################################
# EXPOSE $VNCPORT $JUPYTERPORT


################################################################################
# these go after installs to avoid trivial invalidation


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


# RUN python3 -m pip install transformers datasets deepspeed==0.10.1 ctviewer lightning==2.0.7 more_itertools==10.1.0 tcia_utils==1.6.9



# Set TMP for nvidia build environment
ENV TMP="/tmp"


ENV nnUNet_raw="/home/sliceruser/nnunetMainFolder/nnUNet_raw"
ENV nnUNet_preprocessed="/home/sliceruser/nnunetMainFolder/nnUNet_preprocessed"
ENV nnUNet_results="/home/sliceruser/nnUNet_results"

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
