FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set non-interactive installation to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set environment variables for better compilation support
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
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
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in a more efficient way
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    scikit-learn \
    pandas \
    netCDF4 \
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
    einops \
    && conda install -c conda-forge -c anaconda \
    wrf-python \
    basemap \
    cartopy \
    && conda clean -afy

# Install transformers with a version compatible with PyTorch 2.1
RUN pip install --no-cache-dir \
    transformers==4.35.0

# Install PyTorch compilation dependencies
RUN pip install --no-cache-dir \
    torchviz \
    torchinfo \
    fvcore \
    iopath

# Install KNN_CUDA
RUN pip install --no-cache-dir \
    https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Install PointNet2 ops
RUN pip install --no-cache-dir \
    "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Install additional compilation tools (minimal approach)
RUN pip install --no-cache-dir \
    ninja \
    cython

# Verify PyTorch compilation capabilities
# RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print('Torch compilation test:'); torch.zeros(1).cuda(); print('CUDA working!')"

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