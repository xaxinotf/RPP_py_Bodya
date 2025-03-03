#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
import omp_pi


def main():
    N_values = [5_000_000, 10_000_000, 20_000_000]

    print("=" * 70)
    print("PYBIND11 OPENMP PI CALCULATION".center(70))
    print("Method #1: Rectangles | Method #2: Quarter Circle")
    print("=" * 70)

    for N in N_values:
        print(f"\n--- Calculation for N = {N} ---")

        # Method 1: Rectangles
        print("\n[Method #1: Rectangles]")
        step = 1.0 / N
        print(f"  Step = {step:.10e}")

        start = time.time()
        pi_serial = omp_pi.compute_pi_rectangles_serial(N)
        time_serial = time.time() - start

        start = time.time()
        pi_parallel = omp_pi.compute_pi_rectangles_openmp(N)
        time_parallel = time.time() - start

        speedup = time_serial / time_parallel if time_parallel > 0 else 1.0
        err_serial = abs(pi_serial - math.pi)
        err_parallel = abs(pi_parallel - math.pi)
        diff = abs(pi_serial - pi_parallel)

        print(f"   (Serial)   pi = {pi_serial:.10f}, time = {time_serial:.4f} s, error = {err_serial:.2e}")
        print(f"   (Parallel) pi = {pi_parallel:.10f}, time = {time_parallel:.4f} s, error = {err_parallel:.2e}")
        print(f"   Difference = {diff:e}")
        print(f"   Speedup (Serial/Parallel) = {speedup:.2f}")
        print(f"   Chas obchyslennia (OpenMP): {time_parallel:.4f} s")

        # Method 2: Quarter Circle
        print("\n[Method #2: Quarter Circle]")
        print(f"  Step = {step:.10e}")

        start = time.time()
        pi_serial = omp_pi.compute_pi_quartercircle_serial(N)
        time_serial = time.time() - start

        start = time.time()
        pi_parallel = omp_pi.compute_pi_quartercircle_openmp(N)
        time_parallel = time.time() - start

        speedup = time_serial / time_parallel if time_parallel > 0 else 1.0
        err_serial = abs(pi_serial - math.pi)
        err_parallel = abs(pi_parallel - math.pi)
        diff = abs(pi_serial - pi_parallel)

        print(f"   (Serial)   pi = {pi_serial:.10f}, time = {time_serial:.4f} s, error = {err_serial:.2e}")
        print(f"   (Parallel) pi = {pi_parallel:.10f}, time = {time_parallel:.4f} s, error = {err_parallel:.2e}")
        print(f"   Difference = {diff:e}")
        print(f"   Speedup (Serial/Parallel) = {speedup:.2f}")
        print(f"   Chas obchyslennia (OpenMP): {time_parallel:.4f} s")

    print("\nFinishing pybind11 OpenMP PI calculations.\n")


if __name__ == "__main__":
    main()
