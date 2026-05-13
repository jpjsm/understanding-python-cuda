import cupy as cp

kernel_code = r"""
extern "C" __global__ void add1(float* x) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    x[i] += 1.0f;
}
"""

mod = cp.RawModule(code=kernel_code)
add1 = mod.get_function("add1")

x = cp.arange(16, dtype=cp.float32)
add1((1,), (16,), (x,))
print("NVRTC works:", x)
