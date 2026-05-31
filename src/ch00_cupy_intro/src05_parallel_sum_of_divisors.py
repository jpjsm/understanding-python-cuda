"""_summary_
# 3. Parallel “sum of divisors” kernel

For a given N, compute the sum of all divisors of N.

This is a beautiful exercise because:

- It's similar to primality testing

- But instead of boolean output, you accumulate values

- You must use atomicAdd

- You must avoid overflow

- You must avoid double-counting symmetric divisors

This forces you to understand:

- atomicAdd

- memory_contention

- branching_costs

And it's a perfect stepping stone toward more advanced number‑theory kernels.
"""

import json
import math

import cupy as cp

kernel_code = r"""
extern "C" __global__ 
void find_divisors(
    long long n, 
    long long* small_divisors, 
    long long* big_divisors,
    int* index,
    long long hi_limit)
{
    long long tid = (long long) threadIdx.x;
    long long block = (long long) blockIdx.x;
    long long block_size = (long long) blockDim.x;

    long long divisor = block * block_size + tid;
    
    if (divisor >= hi_limit) return;
    if(divisor == 0) return;
    
    if ((n % divisor) == 0)
    {
        int inx = atomicAdd(index,1);
        small_divisors[inx] = divisor;
        big_divisors[inx] = n / divisor;
    }
}

extern "C" __global__ 
void find_divisors_sum(
    unsigned long long n, 
    unsigned long long* sum, 
    unsigned long long hi_limit)
{
    unsigned long long tid = (unsigned long long) threadIdx.x;
    unsigned long long block = (unsigned long long) blockIdx.x;
    unsigned long long block_size = (unsigned long long) blockDim.x;

    unsigned long long divisor = block * block_size + tid;
    
    if (divisor >= hi_limit) return;
    if(divisor < 2) return;
    
    if ((n % divisor) == 0)
    {
        atomicAdd(sum, divisor);
        unsigned long long other = n / divisor;
        if (other != divisor)
        {
            atomicAdd(sum, other);
        }
    }
}
"""


mod = cp.RawModule(code=kernel_code)
p_find_divisors = mod.get_function("find_divisors")
p_find_divisors_sum = mod.get_function("find_divisors_sum")

DEBUG = False


def get_divisors(n: int) -> set[int]:
    N = int(abs(n))

    if N < 1:
        raise ValueError(f"`n` must be integer different than zero.")

    if N == 1:
        return {1}

    high_limit = math.ceil(math.sqrt(N)) + 1

    threads = 1024
    blocks = (high_limit + threads - 1) // threads

    small_divisors = cp.zeros(2**16, dtype=cp.int64)
    big_divisors = cp.zeros(2**16, dtype=cp.int64)
    index = cp.array([0], dtype=cp.int32)

    p_find_divisors(
        (blocks,), (threads,), (N, small_divisors, big_divisors, index, high_limit)
    )

    cpu_small_divisors = [int(d) for d in small_divisors.get() if d > 0]
    cpu_big_divisors = [int(d) for d in big_divisors.get() if d > 0]
    all_divisors = set(cpu_small_divisors) | set(cpu_big_divisors)
    sum_divisors = sum(all_divisors)
    return all_divisors


def get_divisors_sum(n: int):
    N = int(abs(n))

    if N < 1:
        raise ValueError(f"`n` must be integer different than zero.")

    if N == 1:
        return 1

    high_limit = math.ceil(math.sqrt(N)) + 1

    threads = 1024
    blocks = (high_limit + threads - 1) // threads

    sum = cp.array([0], dtype=cp.uint64)

    p_find_divisors_sum((blocks,), (threads,), (N, sum, high_limit))

    sum_divisors = sum[0].item() + 1 + N
    return sum_divisors


if __name__ == "__main__":
    with open(
        "composites-with-all-divisors.json", "r", encoding="utf-8"
    ) as infile_json:
        composites_test_cases = json.load(infile_json)

    total_test_cases = len(composites_test_cases)
    sum_failed_tests = 0
    get_divisors_failed_tests = 0
    sum_test_success = True
    get_divisors_test_success = True

    for n, divisors in composites_test_cases:
        expected_divisors_set = set(divisors)
        expected_sum = sum(divisors)

        # Test SUM of divisors
        actual_sum = get_divisors_sum(n)
        if actual_sum != expected_sum:
            print(
                f"[ERROR]«get_divisors_sum({n:28,})» actual != expected: {actual_sum:28,} != {expected_sum:28,}"
            )

            sum_test_success = False
            sum_failed_tests += 1

        # Test Get All Divisors
        actual_divisors = get_divisors(n)
        if actual_divisors != expected_divisors_set:
            print(
                f"[ERROR]«get_divisors({n:28,})» Missing values count: {len(expected_divisors_set-actual_divisors):8,}; Un-expected values count != {len(actual_divisors-expected_divisors_set):8,}"
            )

            get_divisors_test_success = False
            get_divisors_failed_tests += 1

    if sum_failed_tests:
        print(f"SUM Failure ratio: {sum_failed_tests/total_test_cases:.2%}")

    if get_divisors_failed_tests:
        print(
            f"Get Divisors Failure ratio: {get_divisors_failed_tests/total_test_cases:.2%}"
        )

    print(f"Test run complete: {sum_test_success=}, {get_divisors_test_success=}")
