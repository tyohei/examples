import chainermn
from mpi4py import MPI

import common


def _split(arr, n):
    N = len(arr)
    for i in range(n):
        head = (i * N) // n
        tail = ((i + 1) * N) // n
        yield arr[head:tail]
        


def main():
    comm = chainermn.create_communicator('naive')
    print_mpi = common.create_print_mpi(comm.mpi_comm)

    ratio = 0.2
    if comm.size * ratio >= comm.inter_size:
        n_masters = comm.inter_size
        if comm.intra_rank == 0:
            is_master = True
        else:
            is_master = False
    else:
        n_masters = max(int(comm.size * ratio), 1)
        nodes = list(range(comm.inter_size))
        chunked_nodes = list(_split(nodes, n_masters))
        print_mpi(chunked_nodes)
        is_master = False
        for chunk in chunked_nodes:
            if comm.inter_rank == chunk[0] and comm.intra_rank == 0:
                is_master = True

    print_mpi('is master: {}'.format(is_master))


if __name__ == '__main__':
    main()
