// omp_pi.cpp
#include <pybind11/pybind11.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

double compute_pi_rectangles_openmp(long N) {
    double step = 1.0 / N;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * step;
}

double compute_pi_rectangles_serial(long N) {
    double step = 1.0 / N;
    double sum = 0.0;
    for (long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * step;
}

double compute_pi_quartercircle_openmp(long N) {
    double step = 1.0 / N;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += std::sqrt(1.0 - x * x);
    }
    return 4.0 * step * sum;
}

double compute_pi_quartercircle_serial(long N) {
    double step = 1.0 / N;
    double sum = 0.0;
    for (long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += std::sqrt(1.0 - x * x);
    }
    return 4.0 * step * sum;
}

PYBIND11_MODULE(omp_pi, m) {
    m.doc() = "OpenMP-based PI calculation module using pybind11";
    m.def("compute_pi_rectangles_openmp", &compute_pi_rectangles_openmp, "Compute PI using rectangles (OpenMP)");
    m.def("compute_pi_rectangles_serial", &compute_pi_rectangles_serial, "Compute PI using rectangles (Serial)");
    m.def("compute_pi_quartercircle_openmp", &compute_pi_quartercircle_openmp, "Compute PI using quarter circle (OpenMP)");
    m.def("compute_pi_quartercircle_serial", &compute_pi_quartercircle_serial, "Compute PI using quarter circle (Serial)");
}
