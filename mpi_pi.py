#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# Якщо хочете форсувати UTF-8 вивід прямо в скрипті (для Windows):
# import sys
# sys.stdout.reconfigure(encoding='utf-8')

from mpi4py import MPI
import time
import math

def compute_pi_segment_rectangles(start_index, end_index, step):
    """
    Method #1 (Rectangles) - часткове обчислення суми
    pi ~ step * sum(4 / (1 + x^2))
    """
    local_sum = 0.0
    for i in range(start_index, end_index):
        x = (i + 0.5) * step
        local_sum += 4.0 / (1.0 + x * x)
    return local_sum

def compute_pi_segment_quartercircle(start_index, end_index, step):
    """
    Method #2 (Quarter Circle) - часткове обчислення суми
    pi ~ 4 * step * sum( sqrt(1 - x^2) )
    """
    local_sum = 0.0
    for i in range(start_index, end_index):
        x = (i + 0.5) * step
        height = math.sqrt(1.0 - x * x)
        local_sum += height
    return local_sum

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Для прикладу візьмемо кілька значень N, щоб показати детальніший вивід
    N_values = [5_000_000, 10_000_000, 20_000_000]

    # Тільки rank=0 виведе шапку
    if rank == 0:
        print("=" * 70)
        print("MPI PI CALCULATION (Two Methods)".center(70))
        print(f"Total processes = {size}")
        print("Method #1: Rectangles, Method #2: Quarter Circle")
        print("=" * 70)

    for N in N_values:
        # Для кожного N виконаємо обчислення двома методами
        # -------------------------------------------------------------
        # METHOD #1: Rectangles
        # -------------------------------------------------------------
        if rank == 0:
            print(f"\n--- [Method #1: Rectangles] Obchyslennia pi dlia N = {N} ---")
            start_time = time.time()
        else:
            start_time = None

        step = 1.0 / N
        chunk = N // size
        start_index = rank * chunk
        end_index = start_index + chunk if rank != (size - 1) else N

        local_sum_rect = compute_pi_segment_rectangles(start_index, end_index, step)
        total_sum_rect = comm.reduce(local_sum_rect, op=MPI.SUM, root=0)

        if rank == 0:
            pi_approx_rect = total_sum_rect * step
            end_time = time.time()
            elapsed_rect = end_time - start_time
            error_rect = abs(pi_approx_rect - math.pi)

            print(f"   Step = {step:.10e}")
            print(f"   pi_approx = {pi_approx_rect:.10f}")
            print(f"   Error vs math.pi = {error_rect:.2e}")
            print(f"   Chas obchyslennia (MPI, {size} protsesiv): {elapsed_rect:.4f} s")

        # -------------------------------------------------------------
        # METHOD #2: Quarter Circle
        # -------------------------------------------------------------
        if rank == 0:
            print(f"\n--- [Method #2: Quarter Circle] Obchyslennia pi dlia N = {N} ---")
            start_time = time.time()
        else:
            start_time = None

        local_sum_circle = compute_pi_segment_quartercircle(start_index, end_index, step)
        total_sum_circle = comm.reduce(local_sum_circle, op=MPI.SUM, root=0)

        if rank == 0:
            pi_approx_circle = 4.0 * step * total_sum_circle
            end_time = time.time()
            elapsed_circle = end_time - start_time
            error_circle = abs(pi_approx_circle - math.pi)

            print(f"   Step = {step:.10e}")
            print(f"   pi_approx = {pi_approx_circle:.10f}")
            print(f"   Error vs math.pi = {error_circle:.2e}")
            print(f"   Chas obchyslennia (MPI, {size} protsesiv): {elapsed_circle:.4f} s")

    # Завершальний вивід від rank=0
    if rank == 0:
        print("\nFinishing MPI calculations (Two Methods).\n")


if __name__ == "__main__":
    main()
