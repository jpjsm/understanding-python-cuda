"""
Given an array of N elements, add all numbers through parallelization
"""

import time
import numpy as np
from numba import cuda


@cuda.jit
def reduce_sum_kernel(input_data, partial_sums):
    # Shared memory for this block
    smem = cuda.shared.array(1024, dtype=np.int64)

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
array_sizes = [
    1000003,
    5000011,
    10000019,
    20000003,
    50000017,
    100000007,
    250000013,
    500000003,
    750000007,
    1000000007,
]

print(
    f"{'N':>16} | {'Blocks':>16} | {'Actual Value':>24} | {'Expected Value':>24} | {'Values':^10} | {'Allocate':>10} | {'Allocate':>10} | {'  Sum':>10} | {'Copy':>10} | {'Final':>10}"
)
print(
    f"{' ':>16} | {'      ':>16} | {'            ':>24} | {'              ':>24} | {'Match':^10} | {'     CPU':>10} | {'     GPU':>10} | {'array':>10} | {'from':>10} | {'  sum':>10}"
)
print(
    f"{' ':>16} | {'      ':>16} | {'            ':>24} | {'              ':>24} | {'     ':^10} | {'memory':>10} | {'memory':>10} | {'     ':>10} | {' GPU':>10} | {'     ':>10}"
)
print(
    f"{'-------------':>16} + {'-------------':>16} + {'--------------------':>24} + {'--------------------':>24} + {'---------':^10} + {'---------':>10} + {'---------':>10} + {'---------':>10} + {'---------':>10} + {'---------':>10}"
)

for N in array_sizes:
    # N = 1024 * 1024
    threads_per_block = 1024
    blocks = (N + threads_per_block * 2 - 1) // (threads_per_block * 2)
    # blocks = 128

    # Input array initialized with ordinal positions
    start = time.perf_counter_ns()
    h_input = np.arange(N, dtype=np.int64)
    end = time.perf_counter_ns()
    allocate_cpu_memory_ms = (end - start) / 1e6

    # Allocate GPU memory
    start = time.perf_counter_ns()
    d_input = cuda.to_device(h_input)
    d_partial = cuda.device_array(blocks, dtype=np.int64)
    end = time.perf_counter_ns()
    allocate_gpu_memory_ms = (end - start) / 1e6

    # Launch kernel
    start = time.perf_counter_ns()
    reduce_sum_kernel[blocks, threads_per_block](d_input, d_partial)
    end = time.perf_counter_ns()
    sum_array_values_ms = (end - start) / 1e6

    # Copy partial sums back
    start = time.perf_counter_ns()
    h_partial = d_partial.copy_to_host()
    end = time.perf_counter_ns()
    copy_results_from_gpu_ms = (end - start) / 1e6

    # Final reduction on CPU (or launch kernel again)
    start = time.perf_counter_ns()
    total_sum = h_partial.sum()
    end = time.perf_counter_ns()
    calculate_final_sum_ms = (end - start) / 1e6

    # Expected result for verification
    expected = (N * (N - 1)) // 2

    # display progress
    print(
        f"{N:16,} | {blocks:16,} | {total_sum:24,} | {expected:24,} | {'YES' if total_sum == expected else 'NO':^10}"
        + f" | {allocate_cpu_memory_ms:10,.1f}"
        + f" | {allocate_gpu_memory_ms:10,.1f}"
        + f" | {sum_array_values_ms:10,.1f}"
        + f" | {copy_results_from_gpu_ms:10,.1f}"
        + f" | {calculate_final_sum_ms:10,.1f}"
    )
