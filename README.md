# SAC Optimization 
This repository contains an implementation of the **Soft Actor-Critic (SAC)** reinforcement learning algorithm. The original SAC code was written in Python, and it has been subsequently converted to C for better performance. To further improve efficiency, parallelization techniques such as OpenMP, CUDA, and MPI were applied to the C implementation. This README outlines the project structure, execution, and performance comparison of each implementation.

## Overview

* **Original Python Implementation (Baseline):** SAC was initially implemented in Python for simplicity and flexibility. However, for large-scale training tasks, Python’s performance can become a bottleneck.

* **C Implementation:** The Python code was converted to C to improve speed, significantly reducing the execution time for training.

* **OpenMP-Parallelized C Implementation:** OpenMP was used to parallelize compute-intensive functions, resulting in further speedup by utilizing multiple threads.

* **CUDA Implementation:** GPU acceleration using CUDA was implemented to speed up batch processing. Although the speedup was notable, memory transfer overheads limited the gains.

* **MPI Implementation:** MPI was used for distributed computing across multiple processors. This approach resulted in the highest speedup due to parallelization over multiple nodes.

## Performance Comparison

The table below summarizes the time efficiency and performance comparison across different implementations.

| Implementation              | Episodes | Time Taken (Seconds) | Best Score | Speedup (vs Python) |
| --------------------------- | -------- | -------------------- | ---------- | ------------------- |
| **Python (Baseline)**       | 100      | 2343.012             | 112.9      | 1.00×               |
| **C Implementation**        | 10,000   | 242.74               | 118.1      | 9.65×               |
| **OpenMP C Implementation** | 10,000   | 187.83               | 117.6      | 12.47×              |
| **CUDA Implementation**     | 1,000    | 716.7                | 123.4      | 3.27×               |
| **MPI Implementation**      | 10,000   | 41.2                 | 128.8      | 56.88×              |

### Key Observations

* **Python Implementation:** Served as the baseline, taking the longest time (2343 seconds) for 100 episodes.

* **C Implementation:** Significant acceleration was achieved (9.65× speedup), thanks to reduced simulation overhead and improved cache efficiency.

* **OpenMP-Parallelized C Implementation:** The performance improved even further, achieving a speedup of 12.47× by parallelizing compute-heavy tasks, though the improvement is constrained by Amdahl's Law.

* **CUDA Implementation:** GPU acceleration improved throughput for batch processing, but the speedup was limited (3.27×) by memory transfer overheads and suboptimal kernel configurations.

* **MPI Implementation:** By distributing the workload across multiple processors, the MPI version achieved the highest speedup (56.88×). This is ideal for scenarios requiring parallel processing over multiple nodes.

## Profiling

Before implementing the optimizations, the Python code was profiled to identify performance bottlenecks. Functional and line profiling results can be found in the **profiling.pdf** file. These insights helped guide the code optimization process.

