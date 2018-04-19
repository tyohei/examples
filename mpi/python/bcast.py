#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import numpy

import common


def bcast(comm):
    n = 8192

    print_mpi = common.create_print_mpi(comm)

    # Allocate buffer and set value
    if comm.rank == 0:
        buf = numpy.arange(n).astype(numpy.float32)
    else:
        buf = numpy.empty(n).astype(numpy.float32)

    # Broadcast
    print_mpi('B: {}'.format(buf), 1)
    print_mpi('Bcast ...')
    comm.Bcast([buf, MPI.FLOAT], root=0)
    print_mpi('Bcast done')
    print_mpi('A: {}'.format(buf), 1)

    print_mpi('========', 0)

    if comm.rank == 0:
        buf = numpy.arange(n).astype(numpy.float32)
    else:
        buf = numpy.array([])

    # Broadcast
    print_mpi('B: {}'.format(buf), 1)
    print_mpi('Bcast ...')
    buf = comm.bcast(buf, root=0)
    print_mpi('Bcast done')
    print_mpi('A: {}'.format(buf), 1)


def main():
    comm = MPI.COMM_WORLD
    bcast(comm)

if __name__ == '__main__':
    main()

