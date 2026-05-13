import numpy as np
from numba import cuda


@cuda.jit
def reduce_stage1(input_data, partial_sums):
    smem = cuda.shared.array(1024, dtype=np.int64)

    tid = cuda.threadIdx.x
    block = cuda.blockIdx.x
    block_size = cuda.blockDim.x

    i = block * block_size * 2 + tid

    temp = 0
    if i < input_data.size:
        temp += input_data[i]
    if i + block_size < input_data.size:
        temp += input_data[i + block_size]

    smem[tid] = temp
    cuda.syncthreads()

    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial_sums[block] = smem[0]


@cuda.jit
def reduce_stage2(partial_sums, result):
    smem = cuda.shared.array(1024, dtype=np.int64)

    tid = cuda.threadIdx.x
    i = tid

    temp = 0
    if i < partial_sums.size:
        temp = partial_sums[i]

    smem[tid] = temp
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        result[0] = smem[0]


N = 1024 * 1024  # 1_048_576
threads_per_block = 1024
blocks_stage1 = (N + threads_per_block * 2 - 1) // (
    threads_per_block * 2
)  # (1_048_576 + 2048 - 1) // 2048 = 1_050_623 // 2048 = 513

# Input: ordinal positions
h_input = np.arange(N, dtype=np.int64)

d_input = cuda.to_device(h_input)
d_partial = cuda.device_array(blocks_stage1, dtype=np.int64)
d_result = cuda.device_array(1, dtype=np.int64)

# Stage 1: big array → partial sums
reduce_stage1[blocks_stage1, threads_per_block](d_input, d_partial)

# Stage 2: partial sums → single sum
threads_stage2 = 1024  # enough to cover 512 partials
reduce_stage2[1, threads_stage2](d_partial, d_result)

# Copy final result
final_sum = d_result.copy_to_host()[0]

expected = (N * (N - 1)) // 2
print("Final sum:", final_sum)
print("Expected :", expected)
print("Match    :", final_sum == expected)
