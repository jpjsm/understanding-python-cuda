"""
Given an array of N elements, add all numbers through parallelization
"""

import time
import numpy as np
from numba import cuda


@cuda.jit
def reduce_single_kernel(input_data, result):
    # Shared memory = on-chip SRAM inside each SM
    smem = cuda.shared.array(1024, dtype=np.int64)

    tid = cuda.threadIdx.x  # Thread lane inside the block (0–1023)
    block = cuda.blockIdx.x  # Which SM this block will run on
    block_size = cuda.blockDim.x
    grid_size = block_size * 2 * cuda.gridDim.x

    # -----------------------------
    # 1. Grid-stride loop
    # -----------------------------
    # Each thread processes multiple elements spaced across the grid.
    # This keeps all SMs busy even if N >> block_size.
    i = block * block_size * 2 + tid
    temp = 0

    while i < input_data.size:
        # Coalesced load: threads in a warp read consecutive elements
        temp += input_data[i]

        # Second load improves memory bandwidth
        if i + block_size < input_data.size:
            temp += input_data[i + block_size]

        # Jump ahead by the full grid size
        i += grid_size

    # -----------------------------
    # 2. Block-level reduction
    # -----------------------------
    # All threads write their partial sums into shared memory.
    # Shared memory is physically inside the SM and is extremely fast.
    smem[tid] = temp
    cuda.syncthreads()

    # Tree reduction: each round halves the number of active threads.
    # This maps directly to warp execution lanes.
    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # -----------------------------
    # 3. Global accumulation
    # -----------------------------
    # Only one thread per block writes the block's result.
    # Atomic add ensures correctness when multiple SMs update result[0].
    if tid == 0:
        cuda.atomic.add(result, 0, smem[0])


# -------------------------
# Run it
# -------------------------
array_sizes = [
    101,
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

# N = 1024 * 1024
threads = 1024
blocks = 128  # See https://copilot.microsoft.com/shares/oueAgWf39wWv2UGeDNR3p

# CUDA warm-up
h_input = np.arange(max(array_sizes), dtype=np.int64)
d_input = cuda.to_device(h_input)
d_result = cuda.to_device(np.array([0], dtype=np.int64))
reduce_single_kernel[blocks, threads](d_input, d_result)
actual = d_result.copy_to_host()[0]
cuda.synchronize()

# execute tests
for N in array_sizes:
    # Input array initialized with ordinal positions
    start = time.perf_counter_ns()
    h_input = np.arange(N, dtype=np.int64)
    end = time.perf_counter_ns()
    allocate_cpu_memory_ms = (end - start) / 1e6

    # Allocate GPU memory
    start = time.perf_counter_ns()
    d_input = cuda.to_device(h_input)
    d_result = cuda.to_device(np.array([0], dtype=np.int64))
    end = time.perf_counter_ns()
    allocate_gpu_memory_ms = (end - start) / 1e6

    # Launch kernel
    start = time.perf_counter_ns()
    reduce_single_kernel[blocks, threads](d_input, d_result)
    cuda.synchronize()
    end = time.perf_counter_ns()
    sum_array_values_ms = (end - start) / 1e6

    # Copy result back
    start = time.perf_counter_ns()
    actual = d_result.copy_to_host()[0]
    end = time.perf_counter_ns()
    copy_results_from_gpu_ms = (end - start) / 1e6

    expected = (N * (N - 1)) // 2

    if N == 101:
        continue

    print(
        f"{N:16,} | {blocks:16,} | {actual:24,} | {expected:24,} | {'YES' if actual == expected else 'NO':^10}"
        + f" | {allocate_cpu_memory_ms:10,.1f}"
        + f" | {allocate_gpu_memory_ms:10,.1f}"
        + f" | {sum_array_values_ms:10,.1f}"
        + f" | {copy_results_from_gpu_ms:10,.1f}"
    )
