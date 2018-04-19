from socket import gethostname


def create_mpi_print(comm):
    rank = comm.rank
    size = comm.size
    host = gethostname()
    digits = len(str(size - 1))
    prefix = '[{{:0{}}}/{}:{}] '.format(digits, size, host).format(rank)

    def mpi_print(*args, root=None, **kwargs):
        for i in range(size):
            if i == rank:
                if root is not None:
                    if i == root:
                        print(prefix, end='')
                        print(*args, **kwargs)
                else:
                    print(prefix, end='')
                    print(*args, **kwargs)
            comm.Barrier()
    return mpi_print


def main():
    from mpi4py import MPI
    mpi_print = create_mpi_print(MPI.COMM_WORLD)
    mpi_print('Hello World!', 'yes')


if __name__ == '__main__':
    main()
