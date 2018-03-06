from mpi4py import MPI


def create_printhost(comm):
    rank = comm.rank
    size = comm.size
    host = MPI.Get_processor_name()
    def printhost(msg):
        digits = len(str(size - 1))
        prefix = '[{{:0{}}}/{}:{}]: '.format(digits, size, host)
        prefix = prefix.format(rank)
        for i in range(MPI.COMM_WORLD.size):
            if i == MPI.COMM_WORLD.rank:
                print(prefix + msg)
                MPI.COMM_WORLD.Barrier()
    return printhost


def main():
    printhost = create_printhost(MPI.COMM_WORLD)
    printhost('Hello World!')


if __name__ == '__main__':
    main()
