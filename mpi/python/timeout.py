from mpi4py import MPI
import numpy as np
import time

import common


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print_mpi = common.create_print_mpi(comm)

    send_buf = np.array([1, 2], dtype=np.float32)
    print_mpi(send_buf)
    if rank == 1:
        comm.send(send_buf, dest=0, tag=11)

    if rank == 0:
        req = comm.irecv(source=1, tag=11)
        flag, status = req.test()
        if not flag:
            print('sleeping...')
            time.sleep(10)
            print('woke up')
            flag, status = req.test()
            if not flag:
                print('nothing comming..., canceling')
                req.Cancel()
            else:
                print(status)
        else:
            print(status)


if __name__ == '__main__':
    main()
