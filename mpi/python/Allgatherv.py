from mpi4py import MPI
import numpy as np

import common


def create_sendcounts(size, count=8192):
    # Array describing how many elements to send to each process
    sendcounts = []
    sendcount = count
    for i in range(size):
        sendcount = int(sendcount // 2)
        sendcounts.append(sendcount)
    sendcounts[-1] = count - (sum(sendcounts) - sendcounts[-1])
    sendcounts[-1] = 0
    return sendcounts


def create_displs(sendcounts):
    # Array describing the displacements where each segment begins
    displs = []
    head = 0
    for count in sendcounts:
        displs.append(head)
        head += count
    return displs


def create_sendbuf(rank, sendcounts, displs):
    sendbuf = np.arange(displs[rank], displs[rank] + sendcounts[rank],
                        dtype=np.float32)
    return sendbuf


def create_recvbuf(rank, sendcounts, displs):
    recvbuf = np.zeros(sum(sendcounts), dtype=np.float32)
    recvbuf = [recvbuf, sendcounts, displs, MPI.FLOAT]
    return recvbuf


def main():
    comm = MPI.COMM_WORLD
    mpi_print = common.create_mpi_print(comm)

    sendcounts = create_sendcounts(comm.size)
    displs = create_displs(sendcounts)
    mpi_print('sendcounts:', sendcounts)
    mpi_print('displs:', displs)

    sendbuf = create_sendbuf(comm.rank, sendcounts, displs)
    recvbuf = create_recvbuf(comm.rank, sendcounts, displs)
    mpi_print("""
BEFORE: MEAN: {},
        MAX:  {},
        MIN:  {},
        LEN:  {}""".format(
        sendbuf.mean() if len(sendbuf) > 0 else '-',
        sendbuf.max() if len(sendbuf) > 0 else '-',
        sendbuf.min() if len(sendbuf) > 0 else '-',
        len(sendbuf)))
    comm.Allgatherv(sendbuf, recvbuf)
    mpi_print("""
AFTER:  MEAN: {},
        MAX:  {},
        MIN:  {}""".format(
        recvbuf[0].mean(),
        recvbuf[0].max(),
        recvbuf[0].min(),
        len(recvbuf[0])))


if __name__ == '__main__':
    main()

