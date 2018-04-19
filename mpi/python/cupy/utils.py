import socket


def get_intra_rank(comm):
    rank = comm.rank
    host = socket.gethostname()
    sendbuf = (rank, host)
    recvbuf = comm.allgather(sendbuf)

    intra_ranks = []
    for rank_i, host_i in recvbuf:
        if host_i == host:
            intra_ranks.append(rank_i)
    intra_ranks = sorted(intra_ranks)

    intra_rank = intra_ranks.index(rank)
    return intra_rank


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    get_intra_rank(comm)


if __name__ == '__main__':
    main()
