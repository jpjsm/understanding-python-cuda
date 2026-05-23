import math

import cupy as cp

kernel_code = r"""
extern "C" __global__
void is_divisible_odd(long long n, bool* results, long long count)
{
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int block_size = blockDim.x;

    long long i = block * block_size + tid;

    if (i >= count) return;
    
    long long d = 2 * i + 3;
    
    
    results[i] = (n % d) == 0;
    
}

extern "C" __global__
void is_divisible_six(long long n, bool* results, long long count)
{
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int block_size = blockDim.x;

    long long i = block * block_size + tid;

    if (i >= count) return;
    
    bool left  = (n % (6*(i+1) - 1)) == 0;
    bool right = (n % (6*(i+1) + 1)) == 0;
    
    
    results[i] = left || right;
    
}
"""

mod = cp.RawModule(code=kernel_code)
is_divisible_odd = mod.get_function("is_divisible_odd")
is_divisible_six = mod.get_function("is_divisible_six")


def is_prime_number_odd(n):
    N = int(n)
    if N <= 5:
        return N == 2 or N == 3 or N == 5

    if N & 1 == 0:
        return False

    high_limit = math.ceil((math.sqrt(N))) // 2

    threads = 1024
    blocks = (high_limit + threads - 1) // threads

    out_gpu = cp.zeros(high_limit, dtype=cp.bool_)

    is_divisible_odd((blocks,), (threads,), (N, out_gpu, high_limit))

    return not out_gpu.any()


def is_prime_number_six(n):
    if n <= 7:
        return n == 2 or n == 3 or n == 5 or n == 7

    N = int(abs(n))

    if N & 1 == 0 or N % 3 == 0:
        return False

    high_limit = (math.ceil((math.sqrt(N))) // 6) + 1

    threads = 1024
    blocks = (high_limit + threads - 1) // threads

    out_gpu = cp.zeros(high_limit, dtype=cp.bool_)

    is_divisible_six((blocks,), (threads,), (N, out_gpu, high_limit))

    return not out_gpu.any()


if __name__ == "__main__":
    numbers = [
        (1, False),
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (6, False),
        (7, True),
        (8, False),
        (9, False),
        (10, False),
        (11, True),
        (21, False),
        (27, False),
        (39, False),
        (49, False),
        (99, False),
        (1031, True),
        (5003, True),
        (7001, True),
        (10007, True),
        (14143, True),
        (17327, True),
        (19997, True),
        (43331, True),
        (5000011, True),
        (10000019, True),
        (750000007, True),
        (1000000007, True),
        (4294967311, True),
        (1031 * 1031, False),
        (5003 * 5003, False),
        (7001 * 7001, False),
        (10007 * 10007, False),
        (14143 * 14143, False),
        (17327 * 17327, False),
        (19997 * 19997, False),
        (22367 * 22367, False),
        (43331 * 43331, False),
        (5000011 * 5000011, False),
        (10000019 * 10000019, False),
        (750000007 * 750000007, False),
        (1000000007 * 1000000007, False),
        # (4294967311 * 4294967311, False),
    ]

    success_odd = True
    success_six = True
    for n, expected in numbers:
        actual_odd = is_prime_number_odd(n)
        actual_six = is_prime_number_six(n)

        if actual_odd != expected:
            print(
                f"[ERROR: «is_prime_number_odd»] For number {n:,} primality test returned {actual_odd}, expected {expected}"
            )
            success_odd = False

        if actual_six != expected:
            print(
                f"[ERROR: «is_prime_number_six»] For number {n:,} primality test returned {actual_six}, expected {expected}"
            )
            success_six = False

    if success_odd:
        print("Success! «is_prime_number_odd»")

    if success_six:
        print("Success! «is_prime_number_six»")
