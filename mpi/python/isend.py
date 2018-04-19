from mpi4py import MPI
import numpy as np

import common


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if rank == 0:
        send_buf = np.array([1, 2, 3], dtype=np.float32)
        req = comm.isend(send_buf, dest=1)
        req.wait()
    elif rank == 1:
        recv_buf = comm.recv(source=0)
        print(recv_buf)


if __name__ == '__main__':
    main()
