# Author(s): Dr. Patrick Lemoine

import os
import math
import statistics
import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

class Stats:
    def __init__(self, min_, max_, avg, stddev):
        self.min = min_
        self.max = max_
        self.avg = avg
        self.stddev = stddev

def calculate_stats(times):
    """Compute min, max, mean, and standard deviation for a list of times."""
    if not times:
        raise ValueError("Times list is empty")
    min_ = min(times)
    max_ = max(times)
    avg = sum(times) / len(times)
    stddev = statistics.stdev(times) if len(times) > 1 else 0.0
    return Stats(min_, max_, avg, stddev)

def log_stats(world_rank, stats, path):
    """Save stats to file for each MPI rank."""
    with open(os.path.join(path, f"stats_rank_{world_rank}.log"), "w") as logfile:
        logfile.write(f"Rank {world_rank} - Min: {stats.min}, Max: {stats.max}, "
                      f"Avg: {stats.avg}, StdDev: {stats.stddev}\n")

def allocate_shared_matrix(world_size, dtype=np.float32):
    """Allocate a shared array among all MPI ranks (as in C++ with MPI_Win_allocate_shared)."""
    comm = MPI.COMM_WORLD
    itemsize = np.dtype(dtype).itemsize
    shape = (world_size, world_size)
    size = shape[0] * shape[1] * itemsize

    # Only rank 0 allocates the actual memory
    if comm.Get_rank() == 0:
        nbytes = size
    else:
        nbytes = 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
    buf, _ = win.Shared_query(0)  # Get buffer pointer from rank 0
    arr = np.ndarray(shape=shape, dtype=dtype, buffer=buf)
    arr.fill(0)  # Ensure array is zero-initialized

    return arr, win

def run_ping_pong(world_rank, world_size, NbPingPong, QView, QSave, path, file_name):
    """Perform Ping-Pong latency test and record times in a shared array."""
    comm = MPI.COMM_WORLD
    arr, win = allocate_shared_matrix(world_size, dtype=np.float32)

    if QView:
        print(f"NumCPU {os.cpu_count()}")
        print(f"Rank {world_rank}, arr shape = {arr.shape}, world_size = {world_size}")

    NbData = 1
    data = np.zeros(NbData, dtype='i')
    times = []

    for i in range(world_size):
        if i != world_rank:
            Sum = 0.0
            for _ in range(NbPingPong):
                t0 = MPI.Wtime()
                req = comm.Isend([data, MPI.INT], dest=i, tag=0)
                comm.Recv([data, MPI.INT], source=i, tag=0)
                req.Wait()
                dt0 = MPI.Wtime() - t0
                Sum += dt0
                times.append(dt0)
            AvgTime = Sum / NbPingPong
            arr[world_rank, i] = AvgTime  # Write into shared array
            if QView:
                print(f"Process {world_rank} send to process {i}: Avg Time = {AvgTime:.10f}")

    stats = calculate_stats(times)
    log_stats(world_rank, stats, path)

    if QView:
        print(f"Rank {world_rank} - Min: {stats.min}, Max: {stats.max}, "
              f"Avg: {stats.avg}, StdDev: {stats.stddev}")

    # Only Rank 0 saves the full shared matrix to file
    comm.Barrier()
    if world_rank == 0 and QSave:
        np.savetxt(file_name, arr, delimiter=",")

    win.Free()

def print_matrix(world_rank, world_size, QView, file_name):
    """Print out matrix from file."""
    if world_rank == 0 and QView:
        arr = np.loadtxt(file_name, delimiter=",")
        print("\n")
        for i in range(world_size):
            for j in range(world_size):
                print(f"[{i},{j}]={arr[i,j]:.10f}\t", end="")
            print()

def plot_matrix(file_name):
    """Plot the matrix from file."""
    arr = np.loadtxt(file_name, delimiter=",")
    #plt.matshow(arr, cmap='viridis')
    plt.matshow(arr, cmap='RdYlBu_r')  # Bleu -> Jaune -> Rouge
    plt.colorbar()
    plt.title("Average Ping-Pong Times Matrix")
    plt.xlabel("Destination Rank")
    plt.ylabel("Source Rank")
    # Save picture
    base_name = file_name.rsplit('.', 1)[0]
    output_file = f"{base_name}.png"
    plt.savefig(output_file)
    # View
    plt.show()

def save_system_info():
    """Save system CPU and GPU information to file."""
    os.system("lscpu > InfoSystemCPU.txt")
    os.system("lshw -C display > InfoSystemGPU.txt")

def main(path, file_name, num_option, QView, NbPingPong):
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if QView:
        print(f"Nb PingPong={NbPingPong}")

    if num_option == 1:
        run_ping_pong(world_rank, world_size, NbPingPong, QView, True, path, file_name)
        print_matrix(world_rank, world_size, QView, file_name)
        if world_rank == 0:
            plot_matrix(file_name)
    elif num_option == 2 and world_rank == 0:
        save_system_info()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Path', type=str, default='.', help='Path.')
    parser.add_argument('--Name', type=str, default='Matrix.csv', help='Name.')
    parser.add_argument('--NumOption', type=int, default=1, help='NumOption.')
    parser.add_argument('--QView', type=int, default=1, help='QView.')
    parser.add_argument('--NbPing', type=int, default=1, help='NbPing.')
    args = parser.parse_args()

    if not os.path.exists(args.Path):
        os.makedirs(args.Path)

    main(args.Path,
         os.path.join(args.Path, args.Name),
         args.NumOption,
         args.QView,
         args.NbPing)
