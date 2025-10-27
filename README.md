# understanding-python-cuda

A place to begin with CUDA using Python

> _References_:

    - CUDA for Python Programmers: A Beginner’s Guide Using PyCUDA
        - [Part 1 - Introduction to CUDA programming and setting up a development environment using Runpod](https://medium.com/@kachari.bikram42/cuda-for-python-programmers-a-beginners-guide-using-pycuda-596b8bbe041f)
        - [Part 2 - Understanding CUDA architecture and its programming and execution model](https://medium.com/@kachari.bikram42/cuda-for-python-programmers-a-beginners-guide-using-pycuda-part-2-4a82a7453c6d)
        - [Part 3 - An introduction to the CUDA memory model](https://medium.com/@kachari.bikram42/cuda-for-python-programmers-a-beginners-guide-using-pycuda-part-3-36560b30b527)
        - [Part 4 - Introduction to global memory](https://medium.com/@kachari.bikram42/cuda-for-python-programmers-a-beginners-guide-using-pycuda-part-4-d9796f5ae3da)

This is a link: [Part 4 - Introduction to global memory](https://medium.com/@kachari.bikram42/cuda-for-python-programmers-a-beginners-guide-using-pycuda-part-4-d9796f5ae3da)

## Setup

### Conda-forge environment

- `conda create --name understanding-python-cuda python=3.12 ipykernel`
- `conda activate understanding-python-cuda`
- `conda install conda-forge::pycuda`
    >**Note**: Add the Visual Studio C++ compiler to the PATH
    (`nvcc fatal : Cannot find compiler 'cl.exe' in PATH`)
- `conda install conda-forge::cupy`
