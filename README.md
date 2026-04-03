# Parallel Ising Model

This repository contains my final project for the course High Performance Parallel Computing.

The project is based on a 2D Ising model simulation and looks at how much the runtime can be improved by moving from a sequential implementation to parallel versions using CUDA, OpenMP, and hybrid MPI + OpenMP.

The main goal was to speed up the simulation enough that larger problem sizes become more practical to run and compare. For large simulations, a sequential version can quickly become too slow, so the project focuses on how different parallel approaches affect runtime, scaling, and overall performance.

## What is included

The repository contains these implementations:

- `sq_seq.cpp` — sequential CPU baseline version
- `sq_par(uses only OpenMP).cpp` — OpenMP only CPU version
- `sq_par_mpi(Uses MPI and OpenMP).cpp` — hybrid MPI + OpenMP version
- `sq_cuda.cu` — first CUDA version
- `sq_cuda_shared.cu` — CUDA version using shared memory tiles
- `sq_cuda_final.cu` — final CUDA version with further memory-related optimizations

The repository also includes the project report:

- `High_Performance_Parallel_Computing_Final.pdf`

## Project overview

The simulation uses a Metropolis Monte Carlo method on a square lattice with periodic boundary conditions. Spins are updated with an even-odd checkerboard scheme so the parallel versions can update the lattice safely without conflicting updates.

I used this project to compare different ways of parallelising the same simulation:

- Sequential CPU baseline
- CUDA GPU implementation
- OpenMP parallel CPU implementation
- Hybrid MPI + OpenMP implementation for distributed and shared memory parallelism

## Main ideas explored

Some of the main things explored in the project were:

- how to parallelise the sweep updates safely
- how GPU and CPU parallelisation compare on the same problem
- how shared memory affects CUDA performance
- how MPI domain decomposition and ghost cells can be used in the hybrid version
- how profiling can be used to understand performance bottlenecks

## Results

The best result came from the CUDA implementation, which reached about 25.6x speedup over the sequential baseline for the largest benchmark used in the report.

The hybrid MPI + OpenMP version reached about 15.9x speedup, while the OpenMP version reached about 7.3x speedup.

One of the main findings was that the simplest CUDA version performed better than the shared-memory CUDA versions in this case. The project therefore was not only about making the code faster, but also about testing which optimizations actually helped in practice.

## Why this project is useful to show

I think this project is a good example of work in parallel computing because it combines:

- C++ programming
- CUDA
- OpenMP
- MPI
- benchmarking
- profiling
- performance analysis
- comparing multiple implementations of the same problem

It also shows the full process from a sequential baseline to several parallel implementations rather than only presenting one final version.

## Repository structure

A simple structure for the repository is:

```text
.
├── src/
│   ├── sq_seq.cpp
│   ├── sq_par(uses only OpenMP).cpp
│   ├── sq_par_mpi(Uses MPI and OpenMP).cpp
│   ├── sq_cuda.cu
│   ├── sq_cuda_shared.cu
│   └── sq_cuda_final.cu
├── report/
│   └── High_Performance_Parallel_Computing_Final.pdf
├── figures/
│   └── speedup_comparison.png
├── README.md
├── Makefile
└── .gitignore