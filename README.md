# Small tools for HPC with Python

## *MPI Ping-Pong Latency Benchmark*

This program, authored by Dr. Patrick Lemoine, is designed for benchmarking message-passing latency and communication performance between multiple MPI (Message Passing Interface) processes. 
It performs ping-pong latency tests between all MPI ranks, measuring the round-trip communication times and recording detailed statistics.

*Features*

- Uses MPI with Python bindings (mpi4py) for parallel communication across distributed processes.
- Allocates a shared matrix to store average ping-pong times between all pairs of MPI ranks.
- Calculates minimum, maximum, average, and standard deviation of message round-trip times for each rank.
- Logs individual rank statistics to separate files for detailed performance analysis.
- Optionally saves the complete latency matrix to a CSV file and visualizes it as a heatmap using Matplotlib.
- Provides functionality to save system CPU and GPU information.
- Supports configurable number of ping-pong repetitions and optional verbose output for debugging.


## *CPU Activity Monitoring Tool*

This Python program enables detailed monitoring and visualization of CPU usage across multiple cores. 
It offers two operating modes to capture and analyze high-performance computing (HPC) CPU activity over time.

*Features*

- Uses psutil to gather real-time per-core CPU utilization statistics.
- Provides two modes of operation:
        Level 1: Samples CPU usage at regular intervals for a specified number of samples, logging and visualizing the data as a heatmap matrix of CPU activity over time.
        Level 2: Continuously monitors CPU activity with an adjustable time interval and window height, displaying a dynamic scrolling visualization using OpenCV color maps.

- Logs detailed CPU usage per core with timestamps to a rotating log file.
- Visualizes CPU activity matrix using Matplotlib (Level 1) or OpenCV (Level 2) for intuitive performance assessment.
- Supports command-line arguments for flexible runtime configuration including output path, sampling interval, matrix resolution, number of samples, and mode selection.

## *Stat...*

