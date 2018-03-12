from mpi4py import MPI


def create_print_mpi(comm):
    rank = comm.rank
    size = comm.size
    host = MPI.Get_processor_name()
    def print_mpi(msg):
        digits = len(str(size - 1))
        prefix = '[{{:0{}}}/{}:{}]: '.format(digits, size, host)
        prefix = prefix.format(rank)
        for i in range(size):
            if i == rank:
                print(prefix, end='')
                print(msg)
                comm.Barrier()
    return print_mpi


def main():
    printhost = create_printhost(MPI.COMM_WORLD)
    printhost('Hello World!')


if __name__ == '__main__':
    main()
