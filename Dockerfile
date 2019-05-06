FROM general-gpu:latest

# Install some basic utilities
RUN apt-get update && \
    apt-get install  -y --assume-yes apt-utils \
    python3-pip \
    build-essential \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    nano \
    curl \
    vim \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    tmux \
    tar \
    gzip \
 && rm -rf /var/lib/apt/lists/*

ENV HOME /home
WORKDIR $HOME

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=$HOME/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN $HOME/miniconda/bin/conda install conda-build \
 && $HOME/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && $HOME/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=$HOME/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 9.0-specific steps
RUN conda install -y -c pytorch \
    cuda90=1.0 \
    magma-cuda90=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda9.0.176_cudnn7.4.1_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Configure Jupyter notebook to produce Python code each time a notebook is 
# saved
COPY docker_auxiliary docker_auxiliary
RUN jupyter notebook --generate-config && \
    cat docker_auxiliary/jupyter_config_part.txt ~/.jupyter/jupyter_notebook_config.py > config_concatenated.py && \
    mv config_concatenated.py ~/.jupyter/jupyter_notebook_config.py


# Notes 
# ======


