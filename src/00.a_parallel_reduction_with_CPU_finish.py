"""
Given an array of 1_048_576 elements, add all numbers through parallelization
"""

import numpy as np
from numba import cuda


@cuda.jit
def reduce_sum_kernel(input_data, partial_sums):
    # Shared memory for this block
    smem = cuda.shared.array(1024, dtype=np.int32)

    tid = cuda.threadIdx.x
    block = cuda.blockIdx.x
    block_size = cuda.blockDim.x

    # Global index: each thread loads 2 elements
    i = block * block_size * 2 + tid

    # Load into shared memory (tree root)
    temp = 0
    if i < input_data.size:
        temp += input_data[i]
    if i + block_size < input_data.size:
        temp += input_data[i + block_size]

    smem[tid] = temp
    cuda.syncthreads()

    # Reduction tree: halve active threads each step
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Write block result
    if tid == 0:
        partial_sums[block] = smem[0]


# -------------------------
# Run the reduction
# -------------------------
N = 1024 * 1024
threads_per_block = 1024
blocks = (N + threads_per_block * 2 - 1) // (threads_per_block * 2)

# Input array initialized with ordinal positions
h_input = np.arange(N, dtype=np.int32)

# Allocate GPU memory
d_input = cuda.to_device(h_input)
d_partial = cuda.device_array(blocks, dtype=np.int32)

# Launch kernel
reduce_sum_kernel[blocks, threads_per_block](d_input, d_partial)

# Copy partial sums back
h_partial = d_partial.copy_to_host()

# Final reduction on CPU (or launch kernel again)
total_sum = h_partial.sum()

print("Final sum:", total_sum)

# Expected result for verification
expected = (N * (N - 1)) // 2
print("Expected:", expected)
print("Match:", total_sum == expected)
