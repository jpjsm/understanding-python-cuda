"""
This module tests CuPy correct installation.

Code extracted from here: https://documen.tician.de/pycuda/tutorial.html

Date: 2025-10-27
Version: 0.0.1
"""
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import cupy as cp
import numpy.typing as npt


class BenchamrkResults:
    def __init__(
        self,
        side: int,
        cpu_time: timedelta,
        gpu_time: timedelta,
        pilot_time: timedelta | None,
    ):
        self.Side: int = side
        self.Cpu_time: timedelta = cpu_time
        self.Gpu_time: timedelta = gpu_time
        self.Pilot_time: timedelta | None = pilot_time


def timedelta_total_microseconds(delta: timedelta) -> int:
    days = delta.days
    seconds = delta.seconds
    microseconds: int = delta.microseconds
    microseconds += seconds * 1_000_000
    microseconds += days * 24 * 3_600 * 1_000_000
    return microseconds


def timedelta_total_milliseconds(delta: timedelta) -> float:
    days = delta.days
    seconds = delta.seconds
    microseconds = delta.microseconds
    milliseconds: float = microseconds / 1000.0
    milliseconds += seconds * 1000.0
    milliseconds += days * 24.0 * 3600.0 * 1000.0
    return milliseconds


def benchmark_processor(arr, func, argument) -> timedelta:
    start_time = datetime.now()
    func(arr, argument)  # your argument will be broadcasted into a matrix automatically
    finish_time = datetime.now()
    elapsed_time = finish_time - start_time
    return elapsed_time


def gen_matrix(side: int) -> Tuple[npt.NDArray[np.int64], cp.ndarray]:
    array_cpu = np.random.randint(0, 255, size=(side, side))
    array_gpu = cp.asarray(array_cpu)
    return (array_cpu, array_gpu)


def get_times(
    run_pilot, array_cpu, array_gpu
) -> Tuple[timedelta | None, timedelta, timedelta]:
    gpu_pilot_time = None
    if run_pilot:
        gpu_pilot_time = benchmark_processor(array_gpu, cp.add, 1)
    gpu_time = benchmark_processor(array_gpu, cp.add, array_gpu)
    cpu_time = benchmark_processor(array_cpu, np.add, array_cpu)
    return (gpu_pilot_time, gpu_time, cpu_time)


if __name__ == "__main__":
    benchmarkresults = []
    side = 25_000
    for i in range(0, 5):
        array_cpu, array_gpu = gen_matrix(side)
        gpu_pilot_time, gpu_time, cpu_time = get_times(
            True, array_cpu=array_cpu, array_gpu=array_gpu
        )
        benchmarkresults.append(
            BenchamrkResults(
                side=side,
                cpu_time=cpu_time,
                gpu_time=gpu_time,
                pilot_time=gpu_pilot_time,
            )
        )
        gpu_pilot_time_str = (
            f"{timedelta_total_microseconds(gpu_pilot_time):18,}"
            if gpu_pilot_time
            else f'{"-": >18}'
        )
        print(
            f"{side: 8,} | {timedelta_total_microseconds(cpu_time):18,} | {timedelta_total_microseconds(gpu_time):18,} | { gpu_pilot_time_str }"
        )

        side += 5_000
