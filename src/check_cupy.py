"""
To verify if CuPy is installed correctly and functioning, you can perform a series of checks and run a simple test script.

reference: [python how to check cupy is installed correctly and working](https://github.com/jpjsm/understanding-python-cuda/blob/main/docs/python-check-cupy.md)

"""

import cupy as cp

print(cp.__version__)

try:
    # Create a CuPy array on the GPU
    a_gpu = cp.array([1, 2, 3, 4, 5])
    print("CuPy array created on GPU:", a_gpu)

    # Perform a simple operation
    b_gpu = a_gpu * 2
    print("Result of GPU operation:", b_gpu)

    # Check the device of the array
    print("Array device:", a_gpu.device)

    # Get the number of available CUDA devices
    num_devices = cp.cuda.runtime.getDeviceCount()
    print("Number of available CUDA devices:", num_devices)

    if num_devices > 0:
        print("CuPy is installed correctly and working with your GPU.")
    else:
        print("CuPy is installed, but no CUDA devices were found.")

except cp.cuda.runtime.CudaError as e:
    print(f"CuPy encountered a CUDA error: {e}")
    print("This might indicate an issue with your CUDA installation or driver.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("This might indicate a more general installation issue.")
