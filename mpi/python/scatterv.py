from mpi4py import MPI
import numpy as np

import common


def create_sendcounts(size, count=8192):
    # Array describing how many elements to send to each process
    sendcounts = []
    sendcount = count
    for i in range(size):
        sendcount = sendcount // 2
        sendcounts.append(sendcount)
    sendcounts[-1] = count - (sum(sendcounts) - sendcounts[-1])
    return sendcounts


def create_displs(sendcounts):
    # Array describing the displacements where each segment begins
    displs = []
    head = 0
    for count in sendcounts:
        displs.append(head)
        head += count
    return displs


def create_sendbuf(rank, sendcounts, displs, count=8192):
    if rank == 0:
        sendbuf = np.arange(count, dtype=np.float32)
    else:
        sendbuf = np.array([0], dtype=np.float32)
    sendbuf = [sendbuf, sendcounts, displs, MPI.FLOAT]
    return sendbuf


def create_recvbuf(rank, sendcounts, displs):
    recvbuf = np.zeros(sendcounts[rank], dtype=np.float32)
    return recvbuf


def main():
    comm = MPI.COMM_WORLD
    mpi_print = common.create_mpi_print(comm)

    sendcounts = create_sendcounts(comm.size)
    displs = create_displs(sendcounts)
    mpi_print('sendcounts:', sendcounts)
    mpi_print('displs:', displs)

    sendbuf = create_sendbuf(comm.rank, sendcounts, displs) if comm.rank == 0 \
        else None
    recvbuf = create_recvbuf(comm.rank, sendcounts, displs)
    mpi_print("""
BEFORE: MEAN: {},
        MAX:  {},
        MIN:  {},
        LEN:  {}""".format(
        recvbuf.mean(),
        recvbuf.max(),
        recvbuf.min(),
        len(recvbuf)))
    comm.Scatterv(sendbuf, recvbuf, root=0)
    mpi_print("""
AFTER:  MEAN: {},
        MAX:  {},
        MIN:  {}""".format(
        recvbuf.mean(),
        recvbuf.max(),
        recvbuf.min(),
        len(recvbuf)))


if __name__ == '__main__':
    main()
