import chainer.cuda
from chainermn.communicators import _memory_utility
import cupy
from mpi4py import MPI

import common
import utils

def create_sendbuf(nelems, arr, gpu_buf):
    nbytes = nelems * 4
    gpu_buf.assign(nbytes)
    offset = 0
    for x in arr:
        size = x.size * 4
        gpu_buf.from_device(x, size, offset)
        offset += size
    buf = [gpu_buf.buffer(nbytes), MPI.FLOAT]
    return buf


def create_recvbuf(nelems, arr, gpu_buf):
    nbytes = nelems * 4
    gpu_buf.assign(nbytes)
    offset = 0
    for x in arr:
        size = x.size * 4
        gpu_buf.from_device(x, size, offset)
        offset += size
    buf = [gpu_buf.buffer(nbytes), MPI.FLOAT]
    return buf


def unpack(arr, gpu_buf):
    offset = 0
    for x in arr:
        size = x.size * 4
        gpu_buf.to_device(x, size, offset)
        offset += size


def main():
    comm = MPI.COMM_WORLD
    intra_rank = utils.get_intra_rank(comm)
    chainer.cuda.get_device(intra_rank).use()
    mpi_print = common.create_mpi_print(comm)

    nelems = pow(2, 20)
    sendarr = cupy.random.rand(nelems, dtype=cupy.float32)
    recvarr = cupy.zeros((nelems,), dtype=cupy.float32)
    sendbuf_gpu = _memory_utility.DeviceMemory()
    recvbuf_gpu = _memory_utility.DeviceMemory()
    sendbuf = create_sendbuf(nelems, sendarr, sendbuf_gpu)
    recvbuf = create_recvbuf(nelems, recvarr, recvbuf_gpu) if comm.rank == 0 \
        else None

    mpi_print('BEFORE MEAN: {}, MAX: {}, MIN: {}'.format(
        sendarr.mean(), sendarr.max(), sendarr.min()))

    # YOU MUST SYNC BEFORE COMMUNICATION !!!
    chainer.cuda.Stream.null.synchronize()  
    comm.Reduce(sendbuf, recvbuf)

    unpack(recvarr, recvbuf_gpu)
    mpi_print('AFTER MEAN: {}, MAX: {}, MIN: {}'.format(
        recvarr.mean(), recvarr.max(), recvarr.min()))


if __name__ == '__main__':
    main()
