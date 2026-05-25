"""_summary_
# 2. Parallel “count primes in a range”
Given a range [A, B], count how many primes exist.

You already have a primality kernel. Now:

Launch one thread per number

Each thread calls your primality test (device function)

Each thread writes 1 if prime, 0 if not

Reduce the result to a single count

This forces you to understand:

grid_stride_loops

parallel_reduction_basics

kernel_launch_overhead

This is the first time you'll see how GPU parallelism scales across many independent tasks.
"""

import cupy as cp

kernel_code = r"""
extern "C" __device__ long long isqrt(long long n) {
    long long left = 0, right = n, ans = 0;

    while (left <= right) 
    {
        long long mid = (left + right) / 2;
        if (mid * mid <= n) 
        {
            ans = mid;
            left = mid + 1;
        } 
        else 
        {
            right = mid - 1;
        }
    }
    
    return ans;
}

extern "C" __device__ bool is_prime(long long n)
{
    if (n < 2) return false;
    if (n==2 || n==3) return true;

    if ((n & 1) == 0) return false;
    
    if ((n % 3) == 0) return false;
    
    long long high_limit = isqrt(n) + 1;
    long long k = 5;
    while (k < high_limit)
    {
        if ((n % k) == 0) return false;
        
        if ((n % (k+2)) == 0) return false;

        k+=6; 
    }
    
    return true;
}

extern "C" __global__ void find_primes(long long base, char* primes, long long count)
{
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int block_size = blockDim.x;

    long long i = block * block_size + tid;
    
    if (i >= count) return;

    long long n = base + i;
    
    primes[i] = is_prime(n);
}
"""

mod = cp.RawModule(code=kernel_code)
p_find_primes = mod.get_function("find_primes")

DEBUG = False


PRIMES_LESS_100 = {
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
}


def print_if_debug(template: str):
    if DEBUG:
        print(template)


def find_primes_in_range(lo: int, hi: int) -> int:
    print_if_debug(f"[INFO] find_primes_in_range({lo}, {hi})")
    # validate arguments
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo

    if hi < 2:
        return 0

    if lo < 2:
        lo = 2

    range_size = hi - lo + 1

    threads = 1024
    blocks = (range_size + threads - 1) // threads

    primes_out = cp.zeros(range_size, dtype=cp.byte)

    print_if_debug(
        f"[INFO] «find_primes_in_range»: {lo=}, {hi=}, {range_size=}, {threads=}, {blocks=}"
    )
    p_find_primes((blocks,), (threads,), (lo, primes_out, range_size))

    primes_count = cp.sum(primes_out, dtype=cp.int64)

    print_if_debug(f"[INFO] find_primes_in_range({lo}, {hi}): {primes_count}")
    return primes_count


if __name__ == "__main__":
    # DEBUG = True

    range_lo = 1
    range_hi = 100
    print(f"Test ranges [{range_lo}, {range_hi}] ")
    for lo in range(range_lo, range_hi):
        for hi in range(lo, range_hi + 1):
            primes_in_range = set([x for x in range(lo, hi + 1)]) & PRIMES_LESS_100
            expected = len(primes_in_range)
            actual = find_primes_in_range(lo, hi)
            if actual != expected:
                print(
                    f"find_primes_in_range({lo},{hi}): {actual} {'==' if actual == expected else '!='} {expected} | primes in range {sorted(primes_in_range)}"
                )

    high_numbers = [
        (4294964207, 4294964209),
        (4294964489, 4294964491),
        (4294964897, 4294964899),
        (4294965671, 4294965673),
        (4294965839, 4294965841),
    ]
    print(f"Test high numbers {high_numbers} ")

    for lo, hi in high_numbers:
        actual = find_primes_in_range(lo, hi)
        expected = 2
        if actual != expected:
            print(
                f"find_primes_in_range({lo},{hi}): {actual} {'==' if actual == expected else '!='} {expected}"
            )

    DEBUG = True
    lo = 1
    hi = 10000000
    actual = find_primes_in_range(lo, hi)
    expected = 664_579
    if actual != expected:
        print(
            f"find_primes_in_range({lo},{hi}): {actual} {'==' if actual == expected else '!='} {expected}"
        )

    print("Tests finished")
