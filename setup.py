# setup.py
import sys
from setuptools import setup, Extension
import pybind11
import platform

extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    # Для MSVC
    extra_compile_args.append("/openmp")
else:
    # Для GCC/Clang
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")

ext_modules = [
    Extension(
        "omp_pi",
        ["omp_pi.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    ),
]

setup(
    name="omp_pi",
    version="0.1",
    author="Your Name",
    author_email="your_email@example.com",
    description="OpenMP-based PI calculation module using pybind11",
    ext_modules=ext_modules,
    install_requires=["pybind11"],
)
