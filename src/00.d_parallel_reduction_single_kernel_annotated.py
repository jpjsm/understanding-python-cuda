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
N = 1024 * 1024
threads = 1024
blocks = 128

h_input = np.arange(N, dtype=np.int64)

d_input = cuda.to_device(h_input)
d_result = cuda.to_device(np.array([0], dtype=np.int64))

reduce_single_kernel[blocks, threads](d_input, d_result)

final_sum = d_result.copy_to_host()[0]
expected = (N * (N - 1)) // 2

print("Final sum:", final_sum)
print("Expected :", expected)
print("Match    :", final_sum == expected)
