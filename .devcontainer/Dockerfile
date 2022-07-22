FROM nvidia/cuda:11.2.1-base-ubuntu18.04 as base_build

ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=HEAD
ARG argTokenGithub

LABEL maintainer=gvasudevan@google.com
LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

ENV CUDNN_VERSION=8.1.0.77
ENV TF_TENSORRT_VERSION=7.2.2
ENV CUDA=11.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y --no-install-recommends \
        automake \
        build-essential \
        ca-certificates \
        cuda-command-line-tools-11-2 \
        libcublas-11-2 \
        libcublas-dev-11-2 \
        cuda-nvrtc-11-2 \
        cuda-nvrtc-dev-11-2 \
        cuda-nvprune-11-2 \
        cuda-cudart-dev-11-2 \
        libcufft-dev-11-2 \
        libcurand-dev-11-2 \
        libcusolver-dev-11-2 \
        libcusparse-dev-11-2 \
        curl \
        git \
        libfreetype6-dev \
        libtool \
        libcudnn8=${CUDNN_VERSION}-1+cuda${CUDA} \
        libcudnn8-dev=${CUDNN_VERSION}-1+cuda${CUDA} \
        libcurl3-dev \
        libzmq3-dev \
        mlocate \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        pkg-config \
        python-dev \
        software-properties-common \
        swig \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        libgomp1 \
        cmake \
        python3-distutils \
        python-distutils-extra && \
    find /usr/local/cuda-11.2/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v8.a

# # NOTE: libnvinfer uses cuda11.1 versions
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub && \
#     echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"  > /etc/apt/sources.list.d/tensorRT.list && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends libnvinfer7=${TF_TENSORRT_VERSION}-1+cuda11.0 \
#       libnvinfer-dev=${TF_TENSORRT_VERSION}-1+cuda11.0 \
#       libnvinfer-plugin-dev=${TF_TENSORRT_VERSION}-1+cuda11.0 \
#       libnvinfer-plugin7=${TF_TENSORRT_VERSION}-1+cuda11.0 \
#       # TODO: need to verify
#       libnvonnxparsers7=${TF_TENSORRT_VERSION}-1+cuda11.0 \
#       libnvparsers7=${TF_TENSORRT_VERSION}-1+cuda11.0\
#       libnvonnxparsers-dev=${TF_TENSORRT_VERSION}-1+cuda11.0 \
#       libnvparsers-dev=${TF_TENSORRT_VERSION}-1+cuda11.0 && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*;

# Install python 3.7.
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3-pip python3.7-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.7 -m pip install pip --upgrade && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 0

# # Make python3.7 the default python version
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 0

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


# RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
#     python3 get-pip.py && \
#     rm get-pip.py

# RUN pip3 --no-cache-dir install \
#     future>=0.17.1 \
#     grpcio \
#     h5py \
#     keras_applications>=1.0.8 \
#     keras_preprocessing>=1.1.0 \
#     mock \
#     numpy \
#     portpicker \
#     requests \
#     kymatio==0.2.1 \
#     voxelmorph==0.2 \
#     tensorflow_addons==0.17.0 \
#     h5py==3.7.0 \
#     #itk \
#     #SimpleITK>=2.1.1.2 \
#     pandas \
#     comet_ml \
#     #SimpleITK-SimpleElastix \
#     KevinSR==0.1.19 \
#     intensity-normalization[ants] \
#     git+https://github.com/DIAGNijmegen/picai_baseline \
#     git+https://github.com/DIAGNijmegen/picai_prep \
#     git+https://github.com/DIAGNijmegen/picai_eval \
#     git+https://github.com/balbasty/nitorch#egg=nitorch[all] \
#     report_guided_annotation \
#     --ignore-installed six>=1.12.0
# RUN export GITHUB_USER=jakubMitura14

# ENV env_var_name=$argTokenGithub

# RUN export GITHUB_TOKEN=argToken

# RUN git clone https://github.com/neel-dey/Atlas-GAN.git ${HOME}/externalRepos/conditionalAtlasGAN

 
# RUN git config --global user.name "Jakub Mitura"
# RUN git config --global user.email "jakub.mitura14@gmail.com"
# RUN git config -l

# # Set up Bazel
# ENV BAZEL_VERSION 5.1.1
# WORKDIR /
# RUN mkdir /bazel && \
#     cd /bazel && \
#     curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#     curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
#     chmod +x bazel-*.sh && \
#     ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#     cd / && \
#     rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# # Build TensorFlow with the CUDA configuration
# ENV CI_BUILD_PYTHON python
# ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ENV TF_NEED_CUDA 1
# ENV TF_NEED_TENSORRT 1
# ENV TENSORRT_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
# ENV TF_CUDA_VERSION=11.2
# ENV TF_CUDNN_VERSION=8

# # Fix paths so that CUDNN can be found: https://github.com/tensorflow/tensorflow/issues/8264
# WORKDIR /
# RUN mkdir /usr/lib/x86_64-linux-gnu/include/ && \
#   ln -s /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h && \
#   ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so && \
#   ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.${TF_CUDNN_VERSION} /usr/local/cuda/lib64/libcudnn.so.${TF_CUDNN_VERSION}

# RUN mkdir ${HOME}/scripts

# RUN mkdir ${HOME}/externalRepos/elastix
# RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix



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
RUN pip install shutil
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