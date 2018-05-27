import chainer.cuda
from chainermn.communicators import _memory_utility
import cupy
from mpi4py import MPI
import time

import common
import utils


def main():
    comm = MPI.COMM_WORLD
    intra_rank = utils.get_intra_rank(comm)
    chainer.cuda.get_device(intra_rank).use()
    mpi_print = common.create_mpi_print(comm)

    nelems_list = [2, 4, 8, 16, 32, 64, 128, 256]
    nelems_max = nelems_list[-1] * pow(2, 20)

    sendarr = cupy.random.rand(nelems_max, dtype=cupy.float32)
    recvarr = cupy.zeros((nelems_max,), dtype=cupy.float32)
    if comm.rank == 0:
        print('array initialized...')

    sendbuf_gpu = _memory_utility.DeviceMemory()
    sendbuf_gpu.assign(nelems_max * 4)
    recvbuf_gpu = _memory_utility.DeviceMemory()
    recvbuf_gpu.assign(nelems_max * 4)
    if comm.rank == 0:
        print('GPU buffer initialized...')

    utils.pack([sendarr], sendbuf_gpu)
    if comm.rank == 0:
        print('packed...')

    for nelems in nelems_list:
        nelems *=  pow(2, 20)

        sendbuf = [sendbuf_gpu.buffer(nelems * 4), MPI.FLOAT]
        recvbuf = [recvbuf_gpu.buffer(nelems * 4), MPI.FLOAT] if comm.rank == 0 else None

        if comm.rank == 0:
            s_time = time.time()
        # WE MUST SYNC BEFORE COMMUNICATION !!!
        chainer.cuda.Stream.null.synchronize()  
        comm.Reduce(sendbuf, recvbuf, root=0)

        if comm.rank == 0:
            print('COUNT {} MiBytes, TIME {} sec'.format((nelems*4)/pow(2, 20), time.time() - s_time))


if __name__ == '__main__':
    main()
