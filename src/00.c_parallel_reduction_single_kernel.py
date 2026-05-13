import numpy as np
from numba import cuda


@cuda.jit
def reduce_single_kernel(input_data, result):
    # Shared memory for this block
    smem = cuda.shared.array(1024, dtype=np.int64)

    tid = cuda.threadIdx.x
    block = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = block_size * 2 * cuda.gridDim.x

    # Grid-stride loop: each thread processes multiple elements
    i = block * block_size * 2 + tid
    temp = 0

    while i < input_data.size:
        temp += input_data[i]
        if i + block_size < input_data.size:
            temp += input_data[i + block_size]
        i += grid_size

    # Block-level reduction in shared memory
    smem[tid] = temp
    cuda.syncthreads()

    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # One thread per block atomically adds its block sum
    if tid == 0:
        cuda.atomic.add(result, 0, smem[0])


# -------------------------
# Run it
# -------------------------
N = 1024 * 1024
threads_per_block = 1024
blocks = 128  # more than enough; grid-stride loop will cover N

h_input = np.arange(N, dtype=np.int64)

d_input = cuda.to_device(h_input)
d_result = cuda.device_array(1, dtype=np.int64)
d_result[0] = 0  # initialize accumulator

reduce_single_kernel[blocks, threads_per_block](d_input, d_result)

final_sum = d_result.copy_to_host()[0]
expected = (N * (N - 1)) // 2

print("Final sum:", final_sum)
print("Expected :", expected)
print("Match    :", final_sum == expected)
