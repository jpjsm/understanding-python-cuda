import cupy as cp

kernel_code = r"""
extern "C" __global__
void add1(float* x) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    x[i] += .01f;
}
"""

mod = cp.RawModule(code=kernel_code)
add1 = mod.get_function("add1")

grid = (1,)  # Singleton dim (blocks_x,) => 1 block size
block = (16,)  # 16 threads in singleton dimension (threads_x,) => 16 threads
size = grid[0] * block[0] * 2
x = cp.arange(size, dtype=cp.float32)

add1(grid, block, (x,))
print(",".join([f"{y:9.2f}" for y in x]))
