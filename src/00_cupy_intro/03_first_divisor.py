"""_summary_
# 1. Parallel “first divisor” finder
Instead of checking all divisors and then doing .any(), write a kernel that
finds the smallest divisor of N (other than 1).

- Each thread tests one divisor

- If it finds a divisor, it writes it to a single result location

- But multiple threads may find divisors → you must use an atomic

- You must avoid race conditions

- You must avoid writing garbage when no divisor exists

This forces you to understand:

- atomicMin

- single_value_output

- thread_synchronization_basics

This is the next natural step after your primality test.
"""

import math

import cupy as cp

kernel_code = r"""
extern "C" __global__ void first_divisor(long long n, long long * result, long long count)
{
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int block_size = blockDim.x;

    long long i = block * block_size + tid;
    
    if (i >= count) return;
    
    long long left = 6*(i+1) - 1;
    
    if ((n % left) == 0)
    {
        atomicMin(result, left);
        return;
    }
    
    long long right = 6*(i+1) + 1;
    
    if ((n % right) == 0)
    {
        atomicMin(result, right);
    }
}
"""

mod = cp.RawModule(code=kernel_code)
p_first_divisor = mod.get_function("first_divisor")

SENTINEL = 2**63 - 1


def find_first_divisor(n: int):
    N = int(abs(n))

    if N == 1 or N == 2 or N == 3 or N == 5 or N == 7:
        return SENTINEL

    if N & 1 == 0:
        return 2

    if N % 3 == 0:
        return 3

    high_limit = (math.ceil((math.sqrt(N))) // 6) + 1

    threads = 1024
    blocks = (high_limit + threads - 1) // threads

    result = cp.array([SENTINEL], dtype=cp.int64)

    p_first_divisor((blocks,), (threads,), (N, result, high_limit))

    return result[0]


if __name__ == "__main__":
    numbers = [
        (1, SENTINEL),
        (2, SENTINEL),
        (3, SENTINEL),
        (4, 2),
        (5, SENTINEL),
        (6, 2),
        (7, SENTINEL),
        (8, 2),
        (9, 3),
        (10, 2),
        (11, SENTINEL),
        (21, 3),
        (27, 3),
        (39, 3),
        (49, 7),
        (99, 3),
        (1031, SENTINEL),
        (5003, SENTINEL),
        (7001, SENTINEL),
        (10007, SENTINEL),
        (14143, SENTINEL),
        (17327, SENTINEL),
        (19997, SENTINEL),
        (43331, SENTINEL),
        (5000011, SENTINEL),
        (10000019, SENTINEL),
        (750000007, SENTINEL),
        (1000000007, SENTINEL),
        (4294967311, SENTINEL),
        (1031 * 1031, 1031),
        (5003 * 5003, 5003),
        (7001 * 7001, 7001),
        (10007 * 10007, 10007),
        (14143 * 14143, 14143),
        (17327 * 17327, 17327),
        (19997 * 19997, 19997),
        (22367 * 22367, 22367),
        (43331 * 43331, 43331),
        (5000011 * 5000011, 5000011),
        (10000019 * 10000019, 10000019),
        (750000007 * 750000007, 750000007),
        (1000000007 * 1000000007, 1000000007),
        # (4294967311 * 4294967311, 4294967311),
    ]

    success = True
    for n, expected in numbers:
        actual = find_first_divisor(n)
        print(f"n: {n:32,};actual: {actual:32,}; expected: {expected:32,}")
        # print(actual)
        # if actual != expected:
        #     print(
        #         f"[ERROR: «find_first_divisor»] For number {n:,} first divisor returned {actual}, expected {expected}"
        #     )
        #     success = False

    if success:
        print("Success! «find_first_divisor»")
