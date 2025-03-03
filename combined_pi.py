#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import math

# ===================== Numba Methods =====================
from numba import njit, prange, set_num_threads, get_num_threads


@njit(parallel=True)
def numba_parallel_rectangles(N):
    """Numba (Parallel) – Method 1: Rectangles"""
    step = 1.0 / N
    total = 0.0
    for i in prange(N):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step


@njit
def numba_serial_rectangles(N):
    """Numba (Serial) – Method 1: Rectangles"""
    step = 1.0 / N
    total = 0.0
    for i in range(N):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step


@njit(parallel=True)
def numba_parallel_quartercircle(N):
    """Numba (Parallel) – Method 2: Quarter Circle"""
    step = 1.0 / N
    total = 0.0
    for i in prange(N):
        x = (i + 0.5) * step
        total += math.sqrt(1.0 - x * x)
    return 4.0 * step * total


@njit
def numba_serial_quartercircle(N):
    """Numba (Serial) – Method 2: Quarter Circle"""
    step = 1.0 / N
    total = 0.0
    for i in range(N):
        x = (i + 0.5) * step
        total += math.sqrt(1.0 - x * x)
    return 4.0 * step * total


# ===================== MPI Methods =====================
from mpi4py import MPI


def mpi_compute_results(N):
    """
    Compute MPI results for both methods.
    Only rank=0 returns results.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    results = {}
    step = 1.0 / N
    chunk = N // size
    start_index = rank * chunk
    end_index = start_index + chunk if rank != size - 1 else N

    # Method 1: Rectangles
    local_sum_rect = 0.0
    for i in range(start_index, end_index):
        x = (i + 0.5) * step
        local_sum_rect += 4.0 / (1.0 + x * x)
    total_sum_rect = comm.reduce(local_sum_rect, op=MPI.SUM, root=0)
    if rank == 0:
        results['mpi_rectangles'] = total_sum_rect * step

    # Method 2: Quarter Circle
    local_sum_circle = 0.0
    for i in range(start_index, end_index):
        x = (i + 0.5) * step
        local_sum_circle += math.sqrt(1.0 - x * x)
    total_sum_circle = comm.reduce(local_sum_circle, op=MPI.SUM, root=0)
    if rank == 0:
        results['mpi_quartercircle'] = 4.0 * step * total_sum_circle

    return results


# ===================== Pybind11/OpenMP Methods =====================
try:
    import omp_pi

    pybind11_available = True
except ImportError:
    pybind11_available = False


def run_pybind11_methods(N):
    """
    Compute results using pybind11 OpenMP module.
    """
    results = {}
    results['pybind_rect_serial'] = omp_pi.compute_pi_rectangles_serial(N)
    results['pybind_rect_parallel'] = omp_pi.compute_pi_rectangles_openmp(N)
    results['pybind_circle_serial'] = omp_pi.compute_pi_quartercircle_serial(N)
    results['pybind_circle_parallel'] = omp_pi.compute_pi_quartercircle_openmp(N)
    return results


# ===================== Main =====================
def main():
    # Отримання кількості потоків із аргументу (за замовчуванням 4)
    threads = 4
    if len(sys.argv) > 1:
        try:
            threads = int(sys.argv[1])
        except ValueError:
            pass
    set_num_threads(threads)
    actual_threads = get_num_threads()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Список тестових значень N
    N_values = [5_000_000, 10_000_000, 20_000_000]
    results_table = []

    for N in N_values:
        result = {'N': N}

        # ---------- Numba Methods: Method 1 – Rectangles ----------
        start = time.time()
        pi_numba_serial_rect = numba_serial_rectangles(N)
        t_serial_rect = time.time() - start

        start = time.time()
        pi_numba_parallel_rect = numba_parallel_rectangles(N)
        t_parallel_rect = time.time() - start

        speedup_rect = t_serial_rect / t_parallel_rect if t_parallel_rect > 0 else None
        time_diff_rect = t_serial_rect - t_parallel_rect
        err_serial_rect = abs(pi_numba_serial_rect - math.pi)
        err_parallel_rect = abs(pi_numba_parallel_rect - math.pi)

        result['Numba_Rectangles_Serial'] = pi_numba_serial_rect
        result['Time_Numba_Rectangles_Serial'] = t_serial_rect
        result['Error_Numba_Rectangles_Serial'] = err_serial_rect

        result['Numba_Rectangles_Parallel'] = pi_numba_parallel_rect
        result['Time_Numba_Rectangles_Parallel'] = t_parallel_rect
        result['Error_Numba_Rectangles_Parallel'] = err_parallel_rect
        result['Speedup_Numba_Rectangles'] = speedup_rect
        result['TimeDiff_Numba_Rectangles'] = time_diff_rect

        # ---------- Numba Methods: Method 2 – Quarter Circle ----------
        start = time.time()
        pi_numba_serial_circle = numba_serial_quartercircle(N)
        t_serial_circle = time.time() - start

        start = time.time()
        pi_numba_parallel_circle = numba_parallel_quartercircle(N)
        t_parallel_circle = time.time() - start

        speedup_circle = t_serial_circle / t_parallel_circle if t_parallel_circle > 0 else None
        time_diff_circle = t_serial_circle - t_parallel_circle
        err_serial_circle = abs(pi_numba_serial_circle - math.pi)
        err_parallel_circle = abs(pi_numba_parallel_circle - math.pi)

        result['Numba_QuarterCircle_Serial'] = pi_numba_serial_circle
        result['Time_Numba_QuarterCircle_Serial'] = t_serial_circle
        result['Error_Numba_QuarterCircle_Serial'] = err_serial_circle

        result['Numba_QuarterCircle_Parallel'] = pi_numba_parallel_circle
        result['Time_Numba_QuarterCircle_Parallel'] = t_parallel_circle
        result['Error_Numba_QuarterCircle_Parallel'] = err_parallel_circle
        result['Speedup_Numba_QuarterCircle'] = speedup_circle
        result['TimeDiff_Numba_QuarterCircle'] = time_diff_circle

        # ---------- MPI Methods ----------
        mpi_results = mpi_compute_results(N)
        if rank == 0:
            result['MPI_Rectangles'] = mpi_results.get('mpi_rectangles', None)
            result['MPI_QuarterCircle'] = mpi_results.get('mpi_quartercircle', None)

        # ---------- Pybind11/OpenMP Methods ----------
        if pybind11_available:
            start = time.time()
            pi_pybind_rect_serial = omp_pi.compute_pi_rectangles_serial(N)
            t_pybind_serial_rect = time.time() - start

            start = time.time()
            pi_pybind_rect_parallel = omp_pi.compute_pi_rectangles_openmp(N)
            t_pybind_parallel_rect = time.time() - start

            speedup_pybind_rect = t_pybind_serial_rect / t_pybind_parallel_rect if t_pybind_parallel_rect > 0 else None
            time_diff_pybind_rect = t_pybind_serial_rect - t_pybind_parallel_rect
            err_pybind_rect_serial = abs(pi_pybind_rect_serial - math.pi)
            err_pybind_rect_parallel = abs(pi_pybind_rect_parallel - math.pi)

            start = time.time()
            pi_pybind_circle_serial = omp_pi.compute_pi_quartercircle_serial(N)
            t_pybind_serial_circle = time.time() - start

            start = time.time()
            pi_pybind_circle_parallel = omp_pi.compute_pi_quartercircle_openmp(N)
            t_pybind_parallel_circle = time.time() - start

            speedup_pybind_circle = t_pybind_serial_circle / t_pybind_parallel_circle if t_pybind_parallel_circle > 0 else None
            time_diff_pybind_circle = t_pybind_serial_circle - t_pybind_parallel_circle
            err_pybind_circle_serial = abs(pi_pybind_circle_serial - math.pi)
            err_pybind_circle_parallel = abs(pi_pybind_circle_parallel - math.pi)

            result['Pybind_Rectangles_Serial'] = pi_pybind_rect_serial
            result['Time_Pybind_Rectangles_Serial'] = t_pybind_serial_rect
            result['Pybind_Rectangles_Parallel'] = pi_pybind_rect_parallel
            result['Time_Pybind_Rectangles_Parallel'] = t_pybind_parallel_rect
            result['Error_Pybind_Rectangles_Serial'] = err_pybind_rect_serial
            result['Error_Pybind_Rectangles_Parallel'] = err_pybind_rect_parallel
            result['Speedup_Pybind_Rectangles'] = speedup_pybind_rect
            result['TimeDiff_Pybind_Rectangles'] = time_diff_pybind_rect

            result['Pybind_QuarterCircle_Serial'] = pi_pybind_circle_serial
            result['Time_Pybind_QuarterCircle_Serial'] = t_pybind_serial_circle
            result['Pybind_QuarterCircle_Parallel'] = pi_pybind_circle_parallel
            result['Time_Pybind_QuarterCircle_Parallel'] = t_pybind_parallel_circle
            result['Error_Pybind_QuarterCircle_Serial'] = err_pybind_circle_serial
            result['Error_Pybind_QuarterCircle_Parallel'] = err_pybind_circle_parallel
            result['Speedup_Pybind_QuarterCircle'] = speedup_pybind_circle
            result['TimeDiff_Pybind_QuarterCircle'] = time_diff_pybind_circle
        else:
            result['Pybind_Rectangles_Serial'] = None
            result['Time_Pybind_Rectangles_Serial'] = None
            result['Pybind_Rectangles_Parallel'] = None
            result['Time_Pybind_Rectangles_Parallel'] = None
            result['Error_Pybind_Rectangles_Serial'] = None
            result['Error_Pybind_Rectangles_Parallel'] = None
            result['Speedup_Pybind_Rectangles'] = None
            result['TimeDiff_Pybind_Rectangles'] = None

            result['Pybind_QuarterCircle_Serial'] = None
            result['Time_Pybind_QuarterCircle_Serial'] = None
            result['Pybind_QuarterCircle_Parallel'] = None
            result['Time_Pybind_QuarterCircle_Parallel'] = None
            result['Error_Pybind_QuarterCircle_Serial'] = None
            result['Error_Pybind_QuarterCircle_Parallel'] = None
            result['Speedup_Pybind_QuarterCircle'] = None
            result['TimeDiff_Pybind_QuarterCircle'] = None

        results_table.append(result)

    # Лише rank 0 виводить таблицю результатів
    if rank == 0:
        print("\n" + "=" * 120)
        print("DETAILED PI CALCULATION RESULTS".center(120))
        print("=" * 120)
        header = (
            f"{'N':>10} | {'Method':>25} | {'Variant':>10} | {'Pi':>12} | {'Time (s)':>10} | {'Time Diff (s)':>14} | {'Error':>10} | {'Speedup':>8}"
        )
        print(header)
        print("-" * 120)
        for row in results_table:
            N_val = row['N']
            # Numba Rectangles
            print(
                f"{N_val:10d} | {'Numba Rectangles':25} | {'Serial':10} | {row['Numba_Rectangles_Serial']:12.10f} | {row['Time_Numba_Rectangles_Serial']:10.4f} | {'-':14} | {row['Error_Numba_Rectangles_Serial']:10.2e} | {'-':8}")
            print(
                f"{N_val:10d} | {'Numba Rectangles':25} | {'Parallel':10} | {row['Numba_Rectangles_Parallel']:12.10f} | {row['Time_Numba_Rectangles_Parallel']:10.4f} | {row['TimeDiff_Numba_Rectangles']:14.4f} | {row['Error_Numba_Rectangles_Parallel']:10.2e} | {row['Speedup_Numba_Rectangles']:8.2f}")
            # Numba Quarter Circle
            print(
                f"{N_val:10d} | {'Numba QuarterCircle':25} | {'Serial':10} | {row['Numba_QuarterCircle_Serial']:12.10f} | {row['Time_Numba_QuarterCircle_Serial']:10.4f} | {'-':14} | {row['Error_Numba_QuarterCircle_Serial']:10.2e} | {'-':8}")
            print(
                f"{N_val:10d} | {'Numba QuarterCircle':25} | {'Parallel':10} | {row['Numba_QuarterCircle_Parallel']:12.10f} | {row['Time_Numba_QuarterCircle_Parallel']:10.4f} | {row['TimeDiff_Numba_QuarterCircle']:14.4f} | {row['Error_Numba_QuarterCircle_Parallel']:10.2e} | {row['Speedup_Numba_QuarterCircle']:8.2f}")
            # MPI Methods
            mpi_rect = row.get('MPI_Rectangles', None)
            mpi_circle = row.get('MPI_QuarterCircle', None)
            if mpi_rect is not None and mpi_circle is not None:
                print(
                    f"{N_val:10d} | {'MPI Rectangles':25} | {'Parallel':10} | {mpi_rect:12.10f} | {'-':10} | {'-':14} | {'-':10} | {'-':8}")
                print(
                    f"{N_val:10d} | {'MPI QuarterCircle':25} | {'Parallel':10} | {mpi_circle:12.10f} | {'-':10} | {'-':14} | {'-':10} | {'-':8}")
            # Pybind11/OpenMP Methods
            if pybind11_available:
                print(
                    f"{N_val:10d} | {'Pybind Rectangles':25} | {'Serial':10} | {row['Pybind_Rectangles_Serial']:12.10f} | {row['Time_Pybind_Rectangles_Serial']:10.4f} | {'-':14} | {row['Error_Pybind_Rectangles_Serial']:10.2e} | {'-':8}")
                print(
                    f"{N_val:10d} | {'Pybind Rectangles':25} | {'Parallel':10} | {row['Pybind_Rectangles_Parallel']:12.10f} | {row['Time_Pybind_Rectangles_Parallel']:10.4f} | {row['TimeDiff_Pybind_Rectangles']:14.4f} | {row['Error_Pybind_Rectangles_Parallel']:10.2e} | {row['Speedup_Pybind_Rectangles']:8.2f}")
                print(
                    f"{N_val:10d} | {'Pybind QuarterCircle':25} | {'Serial':10} | {row['Pybind_QuarterCircle_Serial']:12.10f} | {row['Time_Pybind_QuarterCircle_Serial']:10.4f} | {'-':14} | {row['Error_Pybind_QuarterCircle_Serial']:10.2e} | {'-':8}")
                print(
                    f"{N_val:10d} | {'Pybind QuarterCircle':25} | {'Parallel':10} | {row['Pybind_QuarterCircle_Parallel']:12.10f} | {row['Time_Pybind_QuarterCircle_Parallel']:10.4f} | {row['TimeDiff_Pybind_QuarterCircle']:14.4f} | {row['Error_Pybind_QuarterCircle_Parallel']:10.2e} | {row['Speedup_Pybind_QuarterCircle']:8.2f}")
            print("-" * 120)
        print("=" * 120)


if __name__ == "__main__":
    main()
