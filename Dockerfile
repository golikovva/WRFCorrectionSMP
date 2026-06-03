# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set non-interactive installation to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set environment variables for better compilation support
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA_EXTENSION=1
# ENV TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7 9.0 10.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH=/opt/conda
ENV USE_CUDA=1
ENV USE_CUDNN=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    wget \
    curl \
    libopenmpi-dev \
    libomp-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN conda config --set channel_priority strict

RUN conda update -n base -c defaults -y conda \
 && conda install -n base -c conda-forge -y mamba \
 && mamba install -n base -c conda-forge -y \
      wrf-python \
 && mamba clean -a -y

# Install Python packages in a more efficient way
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    scikit-learn \
    pandas \
    netCDF4 \
    pygrib \
    matplotlib \
    pendulum \
    scipy \
    optuna \
    jupyter \
    jupyterlab \
    notebook \
    addict \
    pytorch-msssim \
    timm \
    basemap \
    cartopy \
    einops \
    geopandas 

# Install transformers with a version compatible with PyTorch 2.1
RUN pip install --no-cache-dir \
    transformers==4.35.0

# Install PyTorch compilation dependencies
RUN pip install --no-cache-dir \
    torchviz \
    torchinfo \
    fvcore \
    iopath \
    ninja \
    cython

# Install KNN_CUDA
RUN pip install --upgrade --no-cache-dir \
    https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Install PointNet2 ops
# RUN pip install --no-build-isolation --no-cache-dir \
#     "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"


RUN pip install --no-cache-dir --upgrade setuptools wheel pybind11

RUN git clone --depth 1 https://github.com/NVIDIA/torch-harmonics.git /tmp/torch-harmonics && \
    cd /tmp/torch-harmonics && \
    pip install -vvv --no-build-isolation .
RUN pip install --no-cache-dir microsoft-aurora
# Verify PyTorch compilation capabilities

RUN pip install --no-cache-dir clearml
ENV CLEARML_API_HOST=https://api.ml.hybrid-modelling.appliedai.tech
ENV CLEARML_WEB_HOST=https://app.ml.hybrid-modelling.appliedai.tech
ENV CLEARML_FILES_HOST=https://files.ml.hybrid-modelling.appliedai.tech

# credentials (usually passed at runtime instead)
ENV CLEARML_API_ACCESS_KEY=""
ENV CLEARML_API_SECRET_KEY=""

# Create necessary directories
RUN mkdir -p /home/experiments/train_test

# Expose port
EXPOSE 9999

# Set environment variable
ENV NAME vgolikovwrf

# Copy source code
COPY . /home

# Set working directory
WORKDIR /home/experiments/train_test

# Test Torch compilation support
RUN python -c "import torch; import transformers; print(f'PyTorch version: {torch.__version__}'); print(f'Transformers version: {transformers.__version__}'); x = torch.randn(10, 10); print('Basic Torch functionality: OK')"

CMD ["/bin/bash"]