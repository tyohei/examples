from mpi4py import MPI
import numpy as np

import common


def main():
    comm = MPI.COMM_WORLD
    print_mpi = common.create_print_mpi(comm)

    # ================================================================

    if comm.rank == 0:
        send_buf = 0
    else:
        send_buf = 1

    recv_buf = comm.allgather(send_buf)
    print(recv_buf)
    
    # ================================================================

    if comm.rank == 0:
        send_buf = [0, 0, 0]
    else:
        send_buf = [1, 1, 1]

    recv_buf = comm.allgather(send_buf)
    print(recv_buf)

    # ================================================================

    if comm.rank == 0:
        send_buf = np.zeros((2,))
    else:
        send_buf = np.arange(3)

    recv_buf = comm.allgather(send_buf)
    print(recv_buf)

    # ================================================================

    
if __name__ == '__main__':
    main()
