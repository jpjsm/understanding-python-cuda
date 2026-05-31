from datetime import datetime, timedelta
from functools import total_ordering
import heapq
from itertools import combinations, chain
import json
import math
import random
from typing import Iterator

from src02_is_prime import is_prime_number_six as is_prime


@total_ordering
class Divisors:
    def __init__(self, divs):
        if not isinstance(divs, (list, tuple)) and not isinstance(divs, int):
            raise ValueError("divs must be a list, tuple, or single int")

        if isinstance(divs, int):
            divs = [divs]

        if len(divs) == 0 or not all([isinstance(d, int) for d in divs]):
            raise ValueError(
                "divs collection must have elements and all elements must be int"
            )

        if isinstance(divs, tuple):
            divs = list(divs)

        self.divisors = divs

    @property
    def numerator(self):
        return math.prod(self.divisors)

    def __eq__(self, other):
        if not isinstance(other, Divisors):
            raise NotImplementedError()

        return self.numerator == other.numerator

    def __lt__(self, other):
        if not isinstance(other, Divisors):
            raise NotImplementedError()

        return self.numerator < other.numerator

    def __str__(self):
        return f"[{self.numerator}, {list(self.divisors)}]"

    def __repr__(self):
        return f"[{self.numerator}, {list(self.divisors)}]"


def keep_save_top_n(stream, n=20, file_name="top_n.json", timeout_secs=18_000):
    print(f"[INFO]«keep_save_top_n»: begin {n=}, {file_name=}, {timeout_secs=}")
    count = 1
    top = []
    end_datetime = datetime.now() + timedelta(seconds=timeout_secs)
    value = next(stream, None)
    while value and len(top) < n:
        divs = Divisors(value)
        heapq.heappush(top, divs)
        value = next(stream, None)
        with open(file_name, "w", encoding="utf-8") as outfile:
            outfile.write(f"[{','.join([str(t) for t in top])}]")
        count += 1

    if count % 1000 == 0:
        print(f"[INFO]«keep_save_top_n»: {count:40,}", end="\r")

    while value and datetime.now() < end_datetime:
        divs = Divisors(value)
        heapq.heappushpop(top, divs)
        value = next(stream, None)
        with open(file_name, "w", encoding="utf-8") as outfile:
            outfile.write(f"[{','.join([str(t) for t in top])}]")
        count += 1
        if count % 1000 == 0:
            print(f"[INFO]«keep_save_top_n»: {count:40,}", end="\r")

    print(f"\n{count=:,}")
    return top


def generate_sets(
    numbers: list[int],
    min_product: int,
    max_product: int,
    min_elements: int,
    timeout_secs: int = 18_000,
) -> Iterator[tuple[int, ...]]:
    """
    Yield all combinations of `numbers` with at least `min_elements` items
    whose product is strictly greater than `min_products` and less than `max_product`.
    """
    n = len(numbers)
    start = datetime.now()

    for size in range(min_elements, n + 1):
        for combo in combinations(numbers, size):
            if min_product < math.prod(combo) < max_product:
                start = datetime.now()
                yield combo
            else:
                delta = datetime.now() - start
                print(f"[INFO]«generate_sets» {delta.total_seconds():40,.6f}", end="\r")
                if delta.total_seconds() > timeout_secs:
                    print("\nTimeout exceeded!")
                    return None


def generate_composite_from_primes(
    min_product: int = 2**60,
    max_product: int = 2**63,
    min_elements: int = 8,
    composites_to_keep: int = 20,
    gen_sets_timeout_secs: int = 600,
    generation_timeout_secs: int = 1800,
    file_name: str = "top_n_composites_and_primes.json",
):
    print("Generating composites for assignment: Parallel “sum of divisors” kernel")
    prime_divisors = [
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
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        997,
        1009,
        991,
        1013,
        983,
        1019,
        9973,
        10007,
        9967,
        10009,
        9949,
        10037,
        99991,
        100003,
        99989,
        100019,
        99971,
        100043,
        999983,
        1000003,
        999979,
        1000033,
        999961,
        1000037,
        9999991,
        10000019,
        9999973,
        10000079,
        9999971,
        10000103,
    ]
    random.shuffle(prime_divisors)

    sets_gen = generate_sets(
        prime_divisors, min_product, max_product, min_elements, gen_sets_timeout_secs
    )

    composites = keep_save_top_n(
        sets_gen,
        composites_to_keep,
        file_name=file_name,
        timeout_secs=generation_timeout_secs,
    )

    print(f"composites = [{','.join([str(t) for t in composites])}]")


def from_composites_with_primes_to_full_divisors(
    input_file_name: str = "composites-with-prime-factors.json",
    output_file_name: str = "composites-with-all-divisors.json",
):
    with open(input_file_name, "r", encoding="utf-8") as infile_json:
        composite_prime_factors = json.load(infile_json)

    test_cases = []
    for composite, prime_factorss in composite_prime_factors:
        divisor_tuples = chain.from_iterable(
            combinations(prime_factorss, r) for r in range(1, len(prime_factorss) + 1)
        )

        divisors_array = [1] + [math.prod(dt) for dt in divisor_tuples]
        if max(divisors_array) != composite:
            print(
                f"[ERROR]«from_composites_with_primes_to_full_divisors»: N: {composite:,} -> {divisors_array}"
            )
            return

        for divisor in divisors_array:
            if composite % divisor != 0:
                print(
                    f"[ERROR]«from_composites_with_primes_to_full_divisors»: Wrong {divisor=:,} for {composite=:,}, it has a remiender of { composite % divisor }"
                )
                return

        test_cases.append((composite, divisors_array))

    with open(output_file_name, "w", encoding="utf-8") as outfile_json:
        json.dump(test_cases, outfile_json)


def generate_large_composite():
    MAX_LONG = 2**63 - 1
    primes = [
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
    ]
    composite = 1
    divisors_sum = 0
    i = 1
    prime_factors = []
    divisors_array = []
    while True:
        print(f"{i=}, {composite=}, {divisors_sum=}, {prime_factors=}")
        if composite * primes[i] > MAX_LONG:
            break

        composite *= primes[i]
        prime_factors.append(primes[i])

        divisor_tuples = chain.from_iterable(
            combinations(prime_factors, r) for r in range(1, len(prime_factors) + 1)
        )
        divisors_array = [1] + [
            math.prod(dt) for dt in divisor_tuples
        ]  # `1` is a valid divisor not included in the divisor tuples
        divisors_sum = sum(divisors_array)
        if divisors_sum > MAX_LONG:
            composite /= primes[i]
            prime_factors = prime_factors[:-1]
            divisor_tuples = chain.from_iterable(
                combinations(prime_factors, r) for r in range(1, len(prime_factors) + 1)
            )
            divisors_array = [1] + [
                math.prod(dt) for dt in divisor_tuples
            ]  # `1` is a valid divisor not included in the divisor tuples
            divisors_sum = sum(divisors_array)
            break

        i += 1

    return (composite, divisors_sum, prime_factors, divisors_array)


def floor_prime(n: int) -> int:
    n = int(n)
    if n < 2:
        return -1

    if n == 2:
        return 2

    if n == 3:
        return 3

    if n & 1 == 0:
        n -= 1

    while n > 1 and not is_prime(n):
        n -= 2
    return n


def ceiling_prime(n: int) -> int:
    n = int(n)
    if n < 2:
        return -1

    if n == 2:
        return 2

    if n == 3:
        return 3

    if n & 1 == 0:
        n += 1

    high_limit = n + math.ceil(math.pow(n, 0.6))
    while n < high_limit and not is_prime(n):
        n += 1
    return n


if __name__ == "__main__":
    floor_tests = [
        (0, -1),
        (1, -1),
        (2, 2),
        (3, 3),
        (4, 3),
        (5, 5),
        (6, 5),
        (7, 7),
        (8, 7),
        (9, 7),
        (10, 7),
        (11, 11),
        (28, 23),
        (101916, 101891),
        (100926, 100913),
        (2010880, 2010733),
    ]

    ceiling_tests = [
        (0, -1),
        (1, -1),
        (2, 2),
        (3, 3),
        (4, 5),
        (5, 5),
        (6, 7),
        (7, 7),
        (8, 11),
        (9, 11),
        (10, 11),
        (11, 11),
        (28, 29),
        (98642, 98663),
        (104694, 104701),
        (2010734, 2010881),
    ]

    floor_tests_success = True
    for n, expected in floor_tests:
        actual = floor_prime(n)
        if actual != expected:
            print(
                f"[ERROR] «floor_prime» failed for n: {n} => actual: {actual} != {expected}"
            )
            floor_tests_success = False

    if floor_tests_success:
        print("Floor tests succeeded!")

    ceiling_tests_success = True
    for n, expected in ceiling_tests:
        actual = ceiling_prime(n)
        if actual != expected:
            print(
                f"[ERROR] «ceiling_prime» failed for n: {n} => actual: {actual} != {expected}"
            )
            ceiling_tests_success = False

    if ceiling_tests_success:
        print("Ceiling tests succeeded!")

    from_composites_with_primes_to_full_divisors()

    # composite, divisors_sum, prime_factors, divisors_array = generate_large_composite()

    # print(f"N: {composite:28,}, Divisors sum: {divisors_sum:28,}")
    # print(f"Divisors count: {len(divisors_array):,}, Prime factors: {prime_factors}")
