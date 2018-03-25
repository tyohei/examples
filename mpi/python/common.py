from mpi4py import MPI


def create_print_mpi(comm):
    rank = comm.rank
    size = comm.size
    host = MPI.Get_processor_name()
    digits = len(str(size - 1))
    prefix = '[{{:0{}}}/{}:{}] '.format(digits, size, host).format(rank)

    def print_mpi(obj, root=None):
        for i in range(size):
            if i == rank:
                if root is not None:
                    if i == root:
                        print(prefix, end='')
                        print(obj)
                else:
                    print(prefix, end='')
                    print(obj)
            comm.Barrier()
    return print_mpi


def main():
    printhost = create_printhost(MPI.COMM_WORLD)
    printhost('Hello World!')


if __name__ == '__main__':
    main()
