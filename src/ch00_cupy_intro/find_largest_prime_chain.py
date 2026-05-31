from itertools import combinations, chain
import json
import math

MAX_LONG = (1 << 63) - 1
MAX_ULONG = (1 << 64) - 1


def find_largest_prime_chain(limit=MAX_LONG):
    # Set upper bound limit (2^63 - 1)

    # Step 1: Generate a pool of prime numbers
    primes = []
    is_prime = [True] * 300
    for p in range(2, 300):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, 300, p):
                is_prime[i] = False

    # Track the optimal results
    best_prod = 0
    best_sigma = 0
    best_chain = []

    # Step 2: Depth-First Search with pruning
    def dfs(idx, count, current_prod, current_sigma, current_chain):
        nonlocal best_prod, best_sigma, best_chain

        # Base case: We reached the target chain length of 15
        if count == 15:
            if current_prod > best_prod:
                best_prod = current_prod
                best_sigma = current_sigma
                best_chain = list(current_chain)
            return

        # Pruning 1: Not enough primes left in the pool to reach 15
        rem = 15 - count
        if idx + rem > len(primes):
            return

        # Pruning 2: Even the smallest remaining primes would exceed the limit
        min_prod = current_prod
        min_sigma = current_sigma
        for i in range(idx, idx + rem):
            min_prod *= primes[i]
            min_sigma *= primes[i] + 1
        if min_prod >= limit or min_sigma >= limit:
            return

        # Explore combinations
        for i in range(idx, len(primes)):
            p = primes[i]
            next_prod = current_prod * p
            # The sum of divisors for square-free numbers is the product of (p + 1)
            next_sigma = current_sigma * (p + 1)

            # Pruning 3: Exceeds 2^63 - 1 limit
            if next_prod >= limit or next_sigma >= limit:
                break

            current_chain.append(p)
            dfs(i + 1, count + 1, next_prod, next_sigma, current_chain)
            current_chain.pop()

    # Start the recursive search
    dfs(0, 0, 1, 1, [])

    divisor_tuples = chain.from_iterable(
        combinations(best_chain, r) for r in range(1, len(best_chain) + 1)
    )

    divisors_array = [1] + [math.prod(dt) for dt in divisor_tuples]

    # Print results
    print(f"Largest Chain (Length {len(best_chain):4})       : {best_chain}")
    print(f"Product of Primes                 : {best_prod:28,}")
    print(f"Sum of Divisors            (Sigma): {best_sigma:28,}")
    print(f"Count of Divisors (divisors_array): {len(divisors_array):28,}")
    print(f"Limit                             : {limit:28,}")
    print(f"Distance to Limit                 : {limit - best_prod:28,}")

    return (best_prod, best_chain)


if __name__ == "__main__":
    file_name: str = "composites-with-prime-factors.json"
    composite_factors = dict()

    with open(file_name, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    for composite, factors in data:
        composite_factors[composite] = factors

    for limit in [MAX_LONG, MAX_ULONG]:
        composite, factors = find_largest_prime_chain(limit)
        composite_factors[composite] = factors

    data = sorted([(k, v) for k, v in composite_factors.items()], key=lambda d: d[0])
    print(data)

    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile)
