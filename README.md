# understanding-python-cuda

A place to begin with CUDA using Python

## Setup

### Conda-forge environment

- `conda create --name understanding-python-cuda python=3.12 ipykernel`
- `conda activate understanding-python-cuda`
- `conda install conda-forge::pycuda`
    >**Note**: Add the Visual Studio C++ compiler to the PATH
    (`nvcc fatal : Cannot find compiler 'cl.exe' in PATH`)
- `conda install conda-forge::cupy`
