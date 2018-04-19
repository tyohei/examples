from mpi4py import MPI
import numpy as np

import common


def create_buf(count=8192):
    buf = np.arange(count, dtype=np.float32)
    buf = [buf, MPI.FLOAT]
    return buf


def main():
    comm = MPI.COMM_WORLD
    mpi_print = common.create_mpi_print(comm)

    sendbuf = create_buf()
    recvbuf = create_buf()[0] if comm.rank == 0 else None
    buf = recvbuf if comm.rank == 0 else sendbuf
    mpi_print("""
BEFORE: MEAN: {},
        MAX:  {},
        MIN:  {}""".format(
        buf[0].mean(),
        buf[0].max(),
        buf[0].min()))
    comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)
    mpi_print("""
AFTER:  MEAN: {},
        MAX:  {},
        MIN:  {}""".format(
        buf[0].mean(),
        buf[0].max(),
        buf[0].min()))


if __name__ == '__main__':
    main()
