# Parallel Ising Model

C++ high performance parallel computing project focused on speeding up a 2D Ising model simulation using CUDA, OpenMP, and hybrid MPI+OpenMP.

This project was developed as part of a High Performance Parallel Computing course. The goal was to reduce runtime compared with a sequential baseline, since simulations like this can become too slow at larger problem sizes. When runtime grows too much, it becomes harder to run larger experiments/simulations, benchmark different approaches, and explore more demanding workloads in practice.

In this project, the same simulation was implemented and compared across several parallel strategies:

- Sequential CPU baseline
- CUDA GPU implementation
- OpenMP parallel CPU implementation
- Hybrid MPI + OpenMP implementation for distributed and shared memory parallelism

The best result came from the CUDA-v1 version, which achieved about **25.6x speedup** over the sequential baseline at lattice size **L = 2048**. The hybrid OpenMP+MPI version reached **15.9x**, while the OpenMP version reached **7.3x**. Overall, the project showed that GPU parallelization gave the strongest performance for this workload. At the same time, the CPU parallel versions also gave strong improvements and remained relevant when GPU access is limited.

## Project overview

The Ising model is a classical model used to study interacting particles and phase transitions. In this project, the system is simulated with a Metropolis Monte Carlo method on a square lattice with periodic boundary conditions.

A checkerboard-style even-odd update scheme is used so spins can be updated safely in parallel without conflicting with neighbouring updates.

The project focused on three main questions:

1. How much can a sequential simulation be sped up with parallel computing?
2. What tradeoffs appear when using different parallel programming models?
3. Which implementation performs best for larger problem sizes?

## What this project demonstrates

This repository shows experience with:

- C++
- CUDA GPU programming
- OpenMP shared-memory parallelism
- MPI distributed-memory parallelism
- Hybrid MPI + OpenMP parallelization
- Benchmarking and performance analysis
- Profiling and bottleneck investigation
- Scientific computing and simulation workflows

## Implementations

### 1. Sequential version
A sequential baseline implementation of the 2D Ising model.

### 2. CUDA versions
Several CUDA implementations were developed and compared:

- **CUDA-v1**: straightforward GPU parallelization
- **CUDA-v2**: shared-memory tiled version
- **CUDA-v3**: smaller variable types to reduce memory traffic

### 3. OpenMP version
A shared-memory CPU implementation using OpenMP to parallelize the Monte Carlo sweeps and reductions.

### 4. Hybrid OpenMP + MPI version
A distributed-memory + shared-memory implementation where the lattice is split across MPI processes and each process uses OpenMP internally.

This version uses domain decomposition, ghost cells, boundary exchange, and global reductions.

## Key technical ideas

- Metropolis Monte Carlo simulation
- Periodic boundary conditions
- Even-odd checkerboard updates for safe parallel execution
- GPU kernel-based sweeps
- Shared-memory tiling in CUDA
- OpenMP loop parallelization and reductions
- MPI domain decomposition
- Ghost-cell exchange with MPI
- Profiling to identify bottlenecks such as energy evaluation, random number generation, and data gathering

## Results summary

Some of the main results from the project were:

- **CUDA-v1:** about **25.6x** speedup over sequential
- **Hybrid OpenMP + MPI:** about **15.9x** speedup over sequential
- **OpenMP:** about **7.3x** speedup over sequential

The shared-memory CUDA versions did not outperform the simpler CUDA-v1 version in this case. This suggested that the extra shared-memory overhead was not fully justified for this workload.

## Why the project matters

This project is a good example of why parallel computing matters in practice.

Without optimization, computational simulations can become slow enough that:

- larger problem sizes take too long to run
- benchmarking becomes harder and slower
- experimentation becomes more limited
- hardware is used less effectively

By parallelizing the same simulation in several ways, this project shows how runtime can be reduced substantially and how different hardware models affect performance.

## Repository structure

A good structure for this repository is:

```text
.
├── src/
│   ├── sq_seq.cpp
│   ├── sq_cuda.cu
│   ├── openmp_version.cpp
│   ├── hybrid_mpi_openmp.cpp
│   └── ...
├── report/
│   └── High_Performance_Parallel_Computing_Final.pdf
├── figures/
│   ├── speedup_plot.png
│   └── benchmark_plot.png
├── scripts/
│   ├── build.sh
│   ├── run.sh
│   └── job.slurm
├── Makefile
└── README.md
```

If your actual filenames differ, just replace the example names above with the real ones used in your code.

## Build

Add your exact build commands here.

Examples:

```bash
g++ -O3 -fopenmp -o ising_openmp openmp_version.cpp
mpicxx -O3 -fopenmp -o ising_hybrid hybrid_mpi_openmp.cpp
nvcc -O3 -o ising_cuda sq_cuda.cu
```

## Run

Add your exact run commands here.

Examples:

```bash
./ising_seq
./ising_openmp
mpirun -np 4 ./ising_hybrid
./ising_cuda
```

## Report

The full project report is included in the repository and contains the model description, implementation details, profiling discussion, benchmarking results, and comparison across the different parallel versions.

## Note

This project was completed as part of a university course project. If you publish it on GitHub, it is a good idea to keep the report and repository text clear about whether the work was done individually or as part of a group.
