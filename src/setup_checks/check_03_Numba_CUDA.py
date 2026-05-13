from numba import cuda
import numpy as np


@cuda.jit
def add1_numba(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


arr = np.arange(16, dtype=np.float32)
d_arr = cuda.to_device(arr)
add1_numba[1, 16](d_arr)
print("Numba works:", d_arr.copy_to_host())
