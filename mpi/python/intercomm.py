from mpi4py import MPI

import common


def main():
    comm = MPI.COMM_WORLD
    printhost = common.create_printhost(comm)
    printhost('Hello World!')

    if comm.rank == 0 or comm.rank == 1:
        intracomm = comm.Split(color=0, key=comm.rank)
    else:
        intracomm = comm.Split(color=1, key=comm.rank)

    printhost('Hello Intra World!: [{}/{}]'.format(intracomm.rank,
                                                   intracomm.size))

    if comm.rank == 0 or comm.rank == 1:
        remote_leader = 2  # Rank in MPI_COMM_WORLD
        local_leader = 1  # Rank in intracomm
    else:
        remote_leader = 1  # Rank in MPI_COMM_WORLD
        local_leader = 0  # Rank in intracomm

    intercomm = intracomm.Create_intercomm(
        local_leader, MPI.COMM_WORLD, remote_leader)

    printhost('Hello Inter World!: [{}/{}]'.format(intercomm.rank,
                                                   intercomm.size))

    if comm.rank == 0 or comm.rank == 1:
        send_buf = 0
        root = MPI.ROOT if intercomm.rank == local_leader else MPI.PROC_NULL
    else:
        send_buf = 16
        root = 1

    recv_buf = intercomm.reduce(send_buf, root=root)
    print(recv_buf)


if __name__ == '__main__':
    main()
