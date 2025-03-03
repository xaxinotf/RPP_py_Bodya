#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import math

from numba import njit, prange, set_num_threads, get_num_threads

@njit(parallel=True)
def compute_pi_parallel_rectangles(N):
    """
    Method #1 (Rectangles):
    pi ~ step * sum( 4 / (1 + x^2) ), x = (i+0.5)*step
    """
    step = 1.0 / N
    total = 0.0
    for i in prange(N):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step

@njit
def compute_pi_serial_rectangles(N):
    """
    (Serial) Method #1 (Rectangles)
    """
    step = 1.0 / N
    total = 0.0
    for i in range(N):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step

@njit(parallel=True)
def compute_pi_parallel_quartercircle(N):
    """
    Method #2 (Quarter Circle):
    pi ~ 4 * step * sum( sqrt(1 - x^2) ), x = (i+0.5)*step
    """
    step = 1.0 / N
    total = 0.0
    for i in prange(N):
        x = (i + 0.5) * step
        height = math.sqrt(1.0 - x * x)
        total += height
    return 4.0 * step * total

@njit
def compute_pi_serial_quartercircle(N):
    """
    (Serial) Method #2 (Quarter Circle)
    """
    step = 1.0 / N
    total = 0.0
    for i in range(N):
        x = (i + 0.5) * step
        height = math.sqrt(1.0 - x * x)
        total += height
    return 4.0 * step * total

def main():
    # Визначимо кількість потоків для Numba (за замовчанням 4)
    threads = 4
    if len(sys.argv) > 1:
        try:
            threads = int(sys.argv[1])
        except ValueError:
            pass

    # Установимо бажану кількість потоків і подивимось, скільки реально використовується
    set_num_threads(threads)
    actual_threads = get_num_threads()

    print("=" * 70)
    print("NUMBA PI CALCULATION (Two Methods)".center(70))
    print(f"Kilkist' potokiv (requested) -> {threads}, real -> {actual_threads}")
    print("Method #1: Rectangles | Method #2: Quarter Circle")
    print("=" * 70)

    # Список N для тестів
    N_values = [5_000_000, 10_000_000, 20_000_000]

    for N in N_values:
        print(f"\n--- O b c h y s l e n n i a   p i   d l i a   N = {N} ---")

        # ------------------------------------------------
        # METHOD #1: RECTANGLES
        # ------------------------------------------------
        print("\n[Method #1: Rectangles]")
        step = 1.0 / N
        print(f"  Step = {step:.10e}")

        # Однопотокове обчислення
        start_time = time.time()
        pi_serial_rect = compute_pi_serial_rectangles(N)
        time_serial_rect = time.time() - start_time

        # Паралельне обчислення
        start_time = time.time()
        pi_parallel_rect = compute_pi_parallel_rectangles(N)
        time_parallel_rect = time.time() - start_time

        # Похибка від math.pi
        err_serial_rect = abs(pi_serial_rect - math.pi)
        err_parallel_rect = abs(pi_parallel_rect - math.pi)
        # Різниця
        diff_rect = abs(pi_serial_rect - pi_parallel_rect)
        # Приріст швидкодії
        speedup_rect = time_serial_rect / time_parallel_rect if time_parallel_rect > 0 else 1.0

        print(f"   (Serial)   pi = {pi_serial_rect:.10f},   chas = {time_serial_rect:.4f} s, error vs math.pi = {err_serial_rect:.2e}")
        print(f"   (Parallel) pi = {pi_parallel_rect:.10f}, chas = {time_parallel_rect:.4f} s, error vs math.pi = {err_parallel_rect:.2e}")
        print(f"   Riznytsia mizh serial i parallel = {diff_rect:e}")
        print(f"   Speedup (T_serial/T_parallel)    = {speedup_rect:.2f}")
        # ДОДАЄМО АНАЛОГ MPI-рядка виводу для паралельного часу:
        print(f"   Chas obchyslennia (Numba, {actual_threads} threads): {time_parallel_rect:.4f} s")

        # ------------------------------------------------
        # METHOD #2: QUARTER CIRCLE
        # ------------------------------------------------
        print("\n[Method #2: Quarter Circle]")
        print(f"  Step = {step:.10e}")

        # Однопотокове обчислення
        start_time = time.time()
        pi_serial_circle = compute_pi_serial_quartercircle(N)
        time_serial_circle = time.time() - start_time

        # Паралельне обчислення
        start_time = time.time()
        pi_parallel_circle = compute_pi_parallel_quartercircle(N)
        time_parallel_circle = time.time() - start_time

        # Похибка від math.pi
        err_serial_circle = abs(pi_serial_circle - math.pi)
        err_parallel_circle = abs(pi_parallel_circle - math.pi)
        # Різниця
        diff_circle = abs(pi_serial_circle - pi_parallel_circle)
        # Приріст швидкодії
        speedup_circle = time_serial_circle / time_parallel_circle if time_parallel_circle > 0 else 1.0

        print(f"   (Serial)   pi = {pi_serial_circle:.10f},   chas = {time_serial_circle:.4f} s, error vs math.pi = {err_serial_circle:.2e}")
        print(f"   (Parallel) pi = {pi_parallel_circle:.10f}, chas = {time_parallel_circle:.4f} s, error vs math.pi = {err_parallel_circle:.2e}")
        print(f"   Riznytsia mizh serial i parallel = {diff_circle:e}")
        print(f"   Speedup (T_serial/T_parallel)    = {speedup_circle:.2f}")
        # ДОДАЄМО АНАЛОГ MPI-рядка виводу для паралельного часу:
        print(f"   Chas obchyslennia (Numba, {actual_threads} threads): {time_parallel_circle:.4f} s")

    print("\nFinishing NUMBA calculations (Two Methods).\n")


if __name__ == "__main__":
    main()
