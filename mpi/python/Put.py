from mpi4py import MPI
import numpy as np

import common


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    print_mpi = common.create_print_mpi(comm)

    if comm.rank == 0:
        buf = np.array([1, 2], dtype=np.float32)
    else:
        buf = np.array([0, 0], dtype=np.float32)

    win = MPI.Win.Create(buf, comm=comm)

    print_mpi(buf)

    window_owner_rank = 1
    win.Fence()
    if comm.rank == 0:
        win.Lock(window_owner_rank)
        win.Put(buf, window_owner_rank)
        win.Unlock(window_owner_rank)
    win.Fence()

    print_mpi(buf)

if __name__ == '__main__':
    main()
