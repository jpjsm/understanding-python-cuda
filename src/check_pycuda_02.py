"""
This module tests PyCuda correct installation.

Code extracted from here: https://documen.tician.de/pycuda/tutorial.html

Date: 2025-10-27
Version: 0.0.1
"""
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

print("started")
a_gpu = gpuarray.to_gpu(numpy.random.randn(4, 4).astype(numpy.float32))
a_doubled = (2 * a_gpu).get()
print(a_doubled)
print(a_gpu)
print("finished")
