from socket import gethostname


def create_print_mpi(comm):
    rank = comm.rank
    size = comm.size
    host = gethostname()
    digits = len(str(size - 1))
    prefix = '[{{:0{}}}/{}:{}] '.format(digits, size, host).format(rank)

    def print_mpi(*args, root=None, **kwargs):
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
    return print_mpi


def main():
    from mpi4py import MPI
    print_mpi = create_print_mpi(MPI.COMM_WORLD)
    print_mpi('Hello World!', 'yes')


if __name__ == '__main__':
    main()
