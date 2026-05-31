import cupy as cp

kernel_code = r"""
extern "C" __global__
void assignindex(long long* input, int N)
{
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int block_size = blockDim.x;

    int i = block * block_size + tid;
    if (i < N)
        input[i]=(long long) i;

}
"""

mod = cp.RawModule(code=kernel_code)
assignindex = mod.get_function("assignindex")


A = cp.zeros(1_000_000, dtype=cp.int64)
N = A.size
threads = 1024
blocks = (N + threads - 1) // threads

print(f"Starting kernel: `assignindex(({blocks:,},),({threads:,},),  (A, N))`")

assignindex((blocks,), (threads,), (A, N))

print("... validating results")
wrong = cp.where(A != cp.arange(N, dtype=cp.int64))[0]

if wrong.size > 0:
    i = int(wrong[0])
    raise ValueError(f"A[{i:,}] = {int(A[i]):,} != {i:,}")

print("success")
