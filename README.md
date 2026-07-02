# Project template: GPU
This repository represents a blueprint for Python projects using pyTorch optimized with NVIDIA GPUs. The image pre-install the following software components:

- Python 3.10
- Anaconda with a default environment
    - pyTorch
    - Linter for appropiate code standards (and config files for reasonable defaults)
        - flake8
        - pyLint
- NVIDIA toolchain
    - CUDA
    - cuBLAS
    - NVIDIA cuDNN
    - NVIDIA NCCL (optimized for NVLink)
    - RAPIDS
    - NVIDIA Data Loading Library (DALI)
    - TensorRT
    - Torch-TensorRT

## Usage
Specify the packages you require in the *requirements.txt*. More complex environment customization goes into *Dockerfile*.

While using Visual Studio Code for development is encouraged, the image does not depend on this IDE in any way. As a side effect, its required server components are not even installed by default if the Dockerfile in root is built manually. Opening the project in VS Code will set the proper default and configure everything appropriately. Alternatively, build the container with `docker build -t <YOUR PROJECT NAME>:0.1 .` and run the container with `docker run -p <YOUR LOCAL PORT>:22 --rm --gpus all <YOUR PROJECT NAME>:0.1`.